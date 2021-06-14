echo "****************** Installing pytorch ******************"
conda install -y pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
apt-get install libturbojpeg
pip install jpeg4py -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom -i https://pypi.org/simple

echo ""
echo ""
echo "****************** Installing vot-toolkit python ******************"
pip install git+https://github.com/votchallenge/vot-toolkit-python

echo ""
echo ""
echo "****************** Installation complete! ******************"
