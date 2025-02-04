<div align="center">

# dlfNet: Cross Layer Refinement Network for Lane Detection

</div>



Pytorch implementation of the paper "[dlfNet: Cross Layer Refinement Network for Lane Detection](https://arxiv.org/abs/2203.10350)" (CVPR2022 Acceptance).

## Introduction
![Arch](.github/arch.png)
- dlfNet exploits more contextual information to detect lanes while leveraging local detailed lane features to improve localization accuracy. 
- dlfNet achieves SOTA result on CULane, Tusimple, and LLAMAS datasets.

## Installation

### Prerequisites
Only test on Ubuntu18.04 and 20.04 with:
- Python >= 3.8 (tested with Python3.8)
- PyTorch >= 1.6 (tested with Pytorch1.6)
- CUDA (tested with cuda10.2)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$dlfNET_ROOT`
```Shell
git clone https://github.com/EADMO/DLFNet.git
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n dlfnet python=3.8 -y
conda activate dlfnet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Or you can install via pip
pip install torch==1.8.0 torchvision==0.9.0

# Install python packages
python setup.py build develop
```

### Data preparation

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $DLFNET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```


#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $DLFNET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```


## Getting Started

### Training
For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/resnet18_culane.py --gpus 0
```

### Validation
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.


## Results
![F1 vs. Latency for SOTA methods on the lane detection](.github/latency_f1score.png)

[assets]: https://github.com/turoad/dlfNet/releases

### CULane

|   Backbone  |  mF1 | F1@50  | F1@75 |
| :---  |  :---:   |   :---:    | :---:|
| [ResNet-18][assets]     | 55.23  |  79.58   | 62.21 |
| [ResNet-34][assets]     | 55.14  |  79.73   | 62.11 |
| [ResNet-101][assets]     | 55.55| 80.13   | 62.96 |
| [DLA-34][assets]     | 55.64|  80.47   | 62.78 |



### TuSimple
|   Backbone   |      F1   | Acc |      FDR     |      FNR   |
|    :---       |          ---:          |       ---:       |       ---:       |      ---:       |
| [ResNet-18][assets]     |    97.89    |   96.84  |    2.28  |  1.92      | 
| [ResNet-34][assets]       |   97.82              |    96.87          |   2.27          |    2.08      | 
| [ResNet-101][assets]      |   97.62|   96.83  |   2.37   |  2.38  |




## Acknowledgement
<!--ts-->
* [Turoad/clrnet](https://github.com/Turoad/clrnet)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->