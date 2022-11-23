# Towards Category-level 6D Pose Estimation for Human-to-Robot Handovers of Handheld Containers for Food and Drinks

### eXplainable AI - feature branch

xxx

The code is adapted from [Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation](https://github.com/hughw19/NOCS_CVPR2019/).
Open-Images is the default branch.

## Table of Contents

1. [Installation](#installation)
    1. [Requirements](#requirements)
    2. [Instructions](#instructions)
2. [Known issues](#known-issues)
3. [Enquiries, Question and Comments](#enquiries-question-and-comments)
4. [Licence](#licence)

## Installation


### Requirements

* Hardware:
    - CUDA 10.0 & cuDNN 7.41

* Software/libraries:   
    - Python 3.5
    - Tensorflow 1.14.0
    - Keras 2.3.0
    - Anaconda/Miniconda
    - libffi-dev
    - Open3D


### Instructions

Preliminary:
```
sudo apt-get update
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
```

1. Install Anaconda or Miniconda (please follow: https://docs.conda.io/en/latest/miniconda.html#linux-installers)


2. Setup the conda environment
```
conda create --name CoPE python=3.5
conda activate CoPE

pip install --upgrade pip

pip install tensorflow=1.14 keras==2.3.0

```

Additional dependencis
```
python3.5 -m pip install opencv-python moviepy open3d scipy scikit-image cython "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

3. Verify install with CPU
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

```

See also tensorflow installation guide [here](https://www.tensorflow.org/install/pip).



## Running a demo
Download the trained model from here.

Single RGB image
```
python demo.py --model_type C --ckpt_path <path-to-trained-model> --rgb ./000001.png
```

RGB video
```
python demo.py --model_type C --ckpt_path <path-to-trained-model> --model_type C --video ./sample_video.mp4
```

## Arguments

### Model modes
| ID | COORDS |
|----|--------|
|  0 |    -   |
|  1 |    X   |

* Model 0 should correspond to Mask R-CNN in principle
* Model 1 should correspond to NOCS in principle


## Training
```
# Train a new model from pretrained COCO weight
python3 train.py
```

## Known issues

* Python 3.5 reached the end of its life on September 13th, 2020 [DEPRECATION]


## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, or you would like to file a bug report or a feature request, use the Github issue tracker. 


## Licence

This work is licensed under the MIT License. To view a copy of this license, see [LICENSE](LICENSE).
