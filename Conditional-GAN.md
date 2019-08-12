# Conditional Generative Adversarial Nets

* **Link:** https://arxiv.org/abs/1411.1784
* **Authors:** Mehdi Mirza, Simon Osindero
* **Year:** 2014

## Summary

A new GAN structure is proposed in which additional data can be fed into the network to generate data coniditioned on the additional input. This allows the GANs to do new tasks such as Image-to-Image translation, Image labeling (multi-modal), etc.

We condition both the generator and discriminator on some additional data y, which can be any kind of auxiliary information such as class labels or data from other modalities. The data y is fed as an input layer to both the generator and the discriminator.

![loss](https://latex.codecogs.com/gif.latex?%5Cunderset%7BG%7D%7Bmin%7D%5Cunderset%7BD%7D%7Bmax%7DV%28D%2CG%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%28x%29%7D%20%5B%5Clog%20%28D%28x%7Cy%29%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bz%20%5Csim%20p_z%28z%29%7D%5Blog%281-D%28G%28z%7Cy%29%29%29%5D)

## Unimodal Results

 MNIST images were conditioned on their class labels encoded as one-hot vectors.

Generator:

* Input layers: 100 dimensional noise vector **z** & label vector **y** were fed into ReLu layers and output is a sigmoid layer.


Discriminator:

* Generated **x** and **y** are fed into maxout layers and then sigmoid layer.

The results are outperformed by many other networks, this serves to be a proof of concept rather than a successful architecture.

The network is evaluated by a Gaussian Parzen window log-likelihood estimate similar to that in the original GAN paper.

## Multimodal Results

Flickr images are labeled with descriptive words.

For the images a convolutional net is trained on the ImageNet dataset with 21,000 labels. The output is used for image representations.

For the word representations a skip-gram model is trained on a corpus of text from the YFCC100M dataset. 

These two networks are left untouched during training.

The generator recieves a noise prior of 100 dimensions maps the image representation + noise to a 200 dimensional word vector.

The discriminator contains ReLu layers and Maxout layers and a Sigmoid layer in the end.
