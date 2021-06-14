# stark-point

### 参数
/experiments/stark_p/baseline.yaml  
/lib/config/stark_p/config.py  
### 模型
/lib/models/stark/stark_point.py  
/lib/models/stark/transformer_point.py  
/lib/train/actors/stark_point.py  
### 修改过的文件
/lib/models/stark/head.py  
/lib/train/data/processing.py  
/lib/train/data/processing_utils.py  
/lib/train/data/sampler.py  
/lib/train/data/transforms.py  
/lib/base_functions.py  

### 训练
```
python tracking/train.py --script stark_p --config baseline --save_dir . --mode multiple --nproc_per_node 8
```
