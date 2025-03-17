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

## Acknowledgement
Many thanks to these excellent open source projects:
 - [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
 - [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion)
 - [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
