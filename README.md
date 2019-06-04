# Semantic Shapes: Custom Semantic Segmentation Tutorial/ Pipeline

![demo](docs/images/demo.gif)

This repository provides a pipeline to develop different semantic segmentation models.

U-Net and FCN-8 are supported.

## Install

Make sure you have a working versions of CUDA, cudnn, and nvidia drivers.

You can see your nvidia driver using `nvidia-smi`
Check your cuda version with `nvcc -V`

I currently use cuda 9.0 with nvidia-driver-418

#### Create an anaconda virtual environment (linux)

```
conda create -n shapes python=3.7
source activate shapes
pip install -r requirements.txt
```

#### pygame

I had some trouble installing pygame on ubuntu 18.04.
I was able to install pygame using:

1. open sources.list
`sudo nano /etc/apt/sources.list`
2. add this line at the bottom on sources.list
`deb http://archive.ubuntu.com/ubuntu/ cosmic-proposed universe`
3. update the list of available software
`sudo apt update`
`sudo apt install python3-pygame`

Windows users can install the precompiled binary.
Just download the 32 or 64 bit whl.

```
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pygame
pip install pygame‑1.9.4‑cp37‑cp37m‑win_amd64.whl
```

#### labelme

labelme is used to create polygon masks over our images
Follow the install found here: https://github.com/wkentaro/labelme

#### pydensecrf

pydensecrf is used for conditional random field post processing.

`pip install git+https://github.com/lucasb-eyer/pydensecrf.git`

If you are on windows, it might be easiest to install from conda.

`https://github.com/lucasb-eyer/pydensecrf/issues/69`

## Collection

**Change imshape in config.py and try importing the models before collecting images**

I used a shape of (256, 256, 3), but make sure before you collect your images\
to make sure they fit with the model you want to use.

Images for FCN-8 should have a shape with a multiple of 32. (in order to be upsampled 32 times)

`python collect.py`

Starts a thread for a webcam (utils.py) as device 0. Press S to save images to the images directory.

## Annotation

Images should be annotated in labelme and output to a separate directory (annotated).
Do not worry about labeling color. The only thing that matters is the polygon class label.

**do not annotate background**

You should only annotate regions of interest with polygons.
If an image is all background, just skip to the next image.
generate_missing_json in utils.py will be called before training if annotations are missing.
It assumes these missing files are all background.
json files will be generated to annotate pure background for those missing annotations.

## Training

config.py will have all the settings you want to change.

1. imshape: (width, height, n_channels)
2. mode: 'binary' or 'multi'
3. model_name: string to name your model's save file
4. logbase: directory for tensorboard. usually: 'logs'
5. hues: dictionary to set hues for each label from json files

`python train.py`

## Tensorboard

Once the model is training, view tensorboard using:

`tensorboard --logdir=logs`

![scalars_tab](docs/images/scalars.png)
![image_tab](docs/images/multi_semantic.png)

## Stream

Stream will use whatever model_name is set in config.py

`python stream.py`

B - Toggle Background

C - Toggle CRF

M - Softmax vs Argmax Mode

## Conditional Random Fields

Conditional random fields may produce an improved mask.

![crf](docs/images/crf.png)
