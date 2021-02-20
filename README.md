# PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models

## Update
[02/2021] We released the training/testing code. The pre-trained model and postprocessing code will be available later.

## Introduction
This repository is code release for PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models (arXiv pdf [here](https://arxiv.org/pdf/1910.06511.pdf)).

In geometry processing, symmetry is a universal type of high-level structural information of 3D models and benefits many geometry processing tasks including shape segmentation, alignment, matching, and completion. Thus it is an important problem to analyze various symmetry forms of 3D shapes. Planar reflective symmetry is the most fundamental one. Traditional methods based on spatial sampling can be time-consuming and may not be able to identify all the symmetry planes. In this paper, we present a novel learning framework to automatically discover global planar reflective symmetry of a 3D shape. Our framework trains an unsupervised 3D convolutional neural network to extract global model features and then outputs possible global symmetry parameters, where input shapes are represented using voxels. We introduce a dedicated symmetry distance loss along with a regularization loss to avoid generating duplicated symmetry planes. Our network can also identify generalized cylinders by predicting their rotation axes. We further provide a method to remove invalid and duplicated planes and axes. We demonstrate that our method is able to produce reliable and accurate results. Our neural network based method is hundreds of times faster than the state-of-the-art methods, which are based on sampling. Our method is also robust even with noisy or incomplete input surfaces.

In this repository, we provide PRS-Net model implementation (with Pytorch) as well as data preparation, training and evaluation scripts on ShapeNet.
## Installation

The code is tested with Ubuntu 18.04, Python 3.8, Pytorch v1.7, TensorFlow v2.4.1, CUDA 10.0 and cuDNN v7.6.

Install the following Python dependencies (with `pip install`):
    
    tensorflow==2.4.1
    tensorboard==2.4.1
    torch==1.7.1
    torchsummary==1.5.1
    scipy==1.6.0

And for running MATLAB code, you need to install [gptoolbox](https://github.com/alecjacobson/gptoolbox).


## Data preprocessing
We use MATLAB to preprocess data. First download [ShapeNetCore.v2](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2/) into `preprocess/shapenet` and then run `preprocess/precomputeShapeData.m`.
## Training
    python train.py --dataroot ./datasets/shapenet --name exp --tf_log --num_quat 3 --num_plane 3 --batchSize 32 --weight 25
## Inference

    python test.py --dataroot ./datasets/shapenet --name exp --num_quat 3 --num_plane 3

## Acknowledgments
The structure of this code is based on [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), and some MATLAB code is based on [volumetricPrimitives](https://github.com/shubhtuls/volumetricPrimitives).

## Citation

If you find our work useful in your research, please consider citing:

    @ARTICLE{9127500,
        author={L. {Gao} and L. -X. {Zhang} and H. -Y. {Meng} and Y. -H. {Ren} and Y. -K. {Lai} and L. {Kobbelt}},
        title={PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models},
        journal={IEEE Transactions on Visualization and Computer Graphics},
        year = {2020},
        volume = {},
        pages = {1-1},
        number = {},
        doi={10.1109/TVCG.2020.3003823}
    }
