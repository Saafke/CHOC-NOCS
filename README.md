# Model code

## A Mixed-Reality Dataset for Category-level 6D Pose and Size Estimation of Hand-occluded Containers

This is the code to run the NOCS model trained on the CHOC mixed-reality dataset. The code is adapted from [Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation](https://github.com/hughw19/NOCS_CVPR2019/).

[[dataset](https://zenodo.org/record/5085801#.Y3zGQ9LP2V4)]
[[webpage](https://corsmal.eecs.qmul.ac.uk/pose.html)]
[[arxiv pre-print](https://arxiv.org/abs/2211.10470)]

## Table of Contents

1. [Installation](#installation)
    1. [Requirements](#requirements)
    2. [Instructions](#instructions)
2. [Running demo](#demo)
3. [Training](#training)
4. [Known issues](#issues)
5. [Enquiries, Question and Comments](#enquiries-question-and-comments)
6. [Licence](#licence)

## Installation <a name="installation"></a>

### Requirements <a name="requirements"></a>

This code has been tested on an Ubuntu 18.04 machine, CUDA 11.6 and CUDNN XXX, with the following libraries.

* Software/libraries:   
    - Python 3.5
    - Tensorflow 1.14.0
    - Keras 2.3.0
    - Anaconda/Miniconda
    - Open3D

### Instructions <a name="instructions"></a>

1. Install the following essentials:
```
sudo apt-get update
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
```

2. Setup the conda environment (optional but strongly recommended):

Install Anaconda or Miniconda (please follow: https://docs.conda.io/en/latest/miniconda.html#linux-installers).
```
conda create --name choc-nocs-env python=3.5
conda activate choc-nocs-env

pip install --upgrade pip

pip install tensorflow-gpu==1.14.0 keras==2.3.0
```

3. Install the following dependencies:
```
python3.5 -m pip install opencv-python moviepy open3d scipy scikit-image cython "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

4. Verify install with CPU
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## Running demo <a name="demo"></a>
You can use the demo to run the model on your own RGB(-D optional) images. First, download the trained model from here (ADD LINK). 

The general command to run the demo is:
```
python demo.py --ckpt_path <path_to_model> --input_folder <path_to_inputs> --pp <post-processing_technique> --draw 
```

Arguments:

- _ckpt\_path_: local path to the trained model (.h5 format)
- _input\_folder_ : local path to the input folder
- _output\_folder_ : local path to the desired output folder; if unspecified it will save in _input\_folder_ > _output_
- _pp_: post-processing technique to compute the 6D pose, _umeyama_ or _epnp_ (default: _umeyama_)
- _draw_: boolean flag to visualise the results

The input folder should be structured as follows (note that depth is optional):

```
input_folder
  |--rgb
  |   |--0001.png
  |   |--0002.png
  |   | ...
  |--depth
  |   |--0001.png
  |   |--0002.png
  |   | ...
```

We provide a sample in [_sample\_folder_](sample_folder).

## Training <a name="training"></a>
```
python3 train.py
```

## Known issues <a name="issues"></a>

* Python 3.5 reached the end of its life on September 13th, 2020 [DEPRECATION]

## Enquiries, Question and Comments <a name="enquiries-question-and-comments"></a>

If you have any further enquiries, question, or comments, or you would like to file a bug report or a feature request, use the Github issue tracker. 

## Licence <a name="license"></a>

This work is licensed under the MIT License. To view a copy of this license, see [LICENSE](LICENSE).
