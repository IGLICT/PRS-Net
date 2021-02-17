# PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models


## Introduction
This repository is code release for PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models (arXiv report [here](https://arxiv.org/pdf/1910.06511.pdf)).

In geometry processing, symmetry is a universal type of high-level structural information of 3D models and benefits many geometry processing tasks including shape segmentation, alignment, matching, and completion. Thus it is an important problem to analyze various symmetry forms of 3D shapes. Planar reflective symmetry is the most fundamental one. Traditional methods based on spatial sampling can be time-consuming and may not be able to identify all the symmetry planes. In this paper, we present a novel learning framework to automatically discover global planar reflective symmetry of a 3D shape. Our framework trains an unsupervised 3D convolutional neural network to extract global model features and then outputs possible global symmetry parameters, where input shapes are represented using voxels. We introduce a dedicated symmetry distance loss along with a regularization loss to avoid generating duplicated symmetry planes. Our network can also identify generalized cylinders by predicting their rotation axes. We further provide a method to remove invalid and duplicated planes and axes. We demonstrate that our method is able to produce reliable and accurate results. Our neural network based method is hundreds of times faster than the state-of-the-art methods, which are based on sampling. Our method is also robust even with noisy or incomplete input surfaces.

In this repository, we provide PRS-Net model implementation (with Pytorch) as well as data preparation, training and evaluation scripts on ShapeNet.

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

## Installation

The code is tested with Ubuntu 18.04, Python 3.8, Pytorch v1.7, TensorFlow v1.15, CUDA 10.0 and cuDNN v7.6.

Install the following Python dependencies (with `pip install`):
    numpy==1.20.1
    torch==1.7.1
    scipy==1.6.0
    

## Inference










