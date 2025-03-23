# OTVIC
OTVIC: A Dataset with Online Transmission for Vehicle-to-Infrastructure Cooperative 3D Object Detection

## News
[2024-06] The paper is accepted by IROS 2024.

## Abstract
Vehicle-to-infrastructure cooperative 3D object detection (VIC3D) is a task that leverages both vehicle and roadside sensors to jointly perceive the surrounding environment. However, considering the high speed of vehicles, the real-time requirements, and the limitations of communication bandwidth, roadside devices transmit the results of perception rather than raw sensor data or feature maps in our real-world scenarios. And affected by various environmental factors, the transmission delay is dynamic. To meet the needs of practical applications, we present OTVIC, which is the first multi-modality and multi-view dataset with online transmission from real scenes for vehicle-to-infrastructure cooperative 3D object detection. The ego-vehicle receives the results of infrastructure perception in real-time, collected from a section of highway in Chengdu, China. Moreover, we propose LfFormer, which is a novel end-to-end multi-modality late fusion framework with transformer for VIC3D task as a baseline based on OTVIC. Experiments prove our fusion framework's effectiveness and robustness.

## Dataset
Download data at [BaiduCloud](https://pan.baidu.com/s/1WrTgJ7wFccfszt15FJwsNA?pwd=f4in) or [OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/zhu_he_zju_edu_cn/Eh9Os8y64hhEqkkGICLTJ9sBKXHWpo39NJ6tg3JArU7AIQ) and organize as follows:
```
data
└── otvic
    ├── annotation
    │   ├── train
    │   ├── val
    │   └── test
    └── data
        ├── can_bus
        ├── image
        │   ├── Backward
        │   ├── Forward
        │   ├── Left
        │   └── Right
        ├── pointcloud
        └── road
```
## Install
```
conda create -n lfformer python=3.8 -y
conda activate otvic
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install requirements.txt -r
conda install -c omgarcia gcc-6 # gcc-6.2
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

## Train and Test
```python
# Please run cam_stream and lidar_stream first to get pretrain models.
./tools/dist_train.sh ./configs/LfFormer/Lfformer_otvic.py 4
./tools/dist_test.sh ./configs/LfFormer/Lfformer_otvic.py ./ckpt/Lfformer_best.pth 4
```

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@inproceedings{zhu2024otvic,
  title={OTVIC: A Dataset with Online Transmission for Vehicle-to-Infrastructure Cooperative 3D Object Detection},
  author={Zhu, He and Wang, Yunkai and Kong, Quyu and Wei, Yufei and Xia, Xunlong and Deng, Bing and Xiong, Rong and Wang, Yue},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={10732--10739},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgement
Many thanks to these excellent open source projects:
 - [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
 - [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion)
 - [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
