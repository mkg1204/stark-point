import torch
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .transformer_point import build_transformer
from .head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh


class STARK_P(nn.Module):
    '''This is the base class for Transformer Tracking'''
    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER", use_gauss=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer_point.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries  [N, C]
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer, used for reduce the channels
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_w, self.feat_h = int(box_head.feat_w), int(box_head.feat_h)
            self.feat_len_s = int(box_head.feat_w * box_head.feat_h)
        self.use_gauss = use_gauss


    def forward(self, img=None, point=None, seq_dict=None, point_embed=None, mode="backbone", gauss_mask=None, run_box_head=True, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(img, point, gauss_mask)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, point_embed, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError


    def forward_backbone(self, input: NestedTensor, point=None, gauss_mask=None):
        # mkg 2021.6.9 Add point as input and generate point query, use gauss_mask to adjust the template feature.
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            point: the relative coordinates of the point, shape [batch_size x 2]
            gauss_mask: gaussian mask centered at the point.    [batch_size x H x W]
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        feat_dict = self.adjust(output_back, pos, gauss_mask)
        # Generate the point query
        point_query = None
        if point is not None:
            bs, _, h, w = pos[-1].shape
            x, y = (w * point[:, 0]).floor().long().clamp(0, w-1), (h * point[:, 1]).floor().long().clamp(0, h-1)     # [bs], [bs]
            index = y * w + x   # [bs]
            b_index = torch.tensor(range(bs), dtype=torch.long, device=index.device)
            feat_emd = feat_dict["feat"][index, b_index]    # [bs, c]
            pos_emd = feat_dict["pos"][index, b_index]      # [bs, c]
            point_query = feat_emd + pos_emd                # [bs, c]
            return feat_dict, point_query
        else:
            return feat_dict
    
    def forward_transformer(self, seq_dict, point_embed, run_box_head=True, run_cls_head=False, need_att_map=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        if need_att_map:
            output_embed, enc_mem, att_maps_t, att_maps_s = self.transformer(seq_dict["feat"], seq_dict["mask"], point_embed, self.query_embed.weight,
                                                                         seq_dict["pos"], return_encoder_output=True, feat_len_s=self.feat_len_s, need_att_map=need_att_map)
        else:
            output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], point_embed, self.query_embed.weight,
                                                     seq_dict["pos"], return_encoder_output=True, feat_len_s=self.feat_len_s, need_att_map=need_att_map)
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed, enc_mem)
        if need_att_map:
            return out, outputs_coord, output_embed, att_maps_t, att_maps_s
        else:
            return out, outputs_coord, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_h, self.feat_w)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord

    def adjust(self, output_back: list, pos_embed: list, gauss_mask=None):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # use gauss mask
        if gauss_mask is not None and self.use_gauss:
            downsampled_gauss = nn.functional.interpolate(gauss_mask.unsqueeze(1), size=tuple(feat.shape[-2:]), mode='bilinear')
            feat = feat + downsampled_gauss
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]

def build_stark_p(cfg, train_flag=True):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg, train_flag)
    model = STARK_P(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model