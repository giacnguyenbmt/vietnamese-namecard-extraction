

## Install MMOCR
# Install MMCV using MIM
pip install -U openmim
mim install mmcv-full
# Install MMDetection
pip install mmdet
# Install MMOCR
# git clone https://github.com/open-mmlab/mmocr.git
cd 'libs/mmocr'
pip uninstall -y opencv-python
pip install -r requirements.txt
pip install -v -e .
cd '../..'

## Install VietOCR
pip install scikit-build easydict
pip install vietocr==0.3.5

