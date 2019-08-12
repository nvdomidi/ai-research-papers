# Improved Techniques for Training GANs

* **Link:** https://arxiv.org/abs/1606.03498
* **Authors:** Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
* **Year:** 2016

## Summary

This paper: 
1. Proposes some techniques that helps stabilize training in models that were previously unstable.
2. Proposes an evaluation metric (Inception score) for comparing the quality of different models.

---

In this paper, the process of training a generative adversarial network is described as the process of finding a Nash Equilibrium - similar to that of game theory – between the generator and the discriminator. A Nash Equilibrium occurs when each player has minimal cost, but traditional gradient based minimization techniques don’t seem to work well on this problem, and there’s no guarantee that this process will converge.

## Techniques that Encourage Convergence

1. **Feature Matching:**

   Instead of directly optimizing on the output of the discriminator, we require the generator to match the statistics of the real data, while the discriminator specifies which statistics are important to match.  By determining a specific intermediate layer of the discriminator, we optimize the L2 distance between the expected features coming from the real data and the features coming from the generated data. 
   
   ![feature matching equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5CVert%20%5Cmathbb%7BE%7D%20_x_%7B%5Csim%7D_%7Bp_%7Bdata%7D%7D%20f%28x%29%20-%20%5Cmathbb%7BE%7D_z_%7B%5Csim%7D_%7Bp_z%28z%29%29%7D%20f%28G%28z%29%29%29%20%5CVert_2%5E2)

   Feature matching has shown to be effective when GAN training becomes unstable.

2. **Minibatch Discrimination:**

   A common failure for GAN is to only learn to emit one point. As the discriminator processes each input one at a time, it is unable to detect this. The gradients of the discriminator can now only push the point produced by the generator around in space, resulting in no improvement.
   This problem can be solved by making the discriminator look at multiple samples at once.

   * Take vector of features of intermediate layer in discriminator ![feature vector](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20f%28x_i%29%20%5Cin%20%5Cmathbb%7BR%7D%5EA)
   * Multiply it by tensor ![tensor](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20T%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BA%20%5Ctimes%20B%20%5Ctimes%20C%7D) to get ![result matrix](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20M_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BB%20%5Ctimes%20C%7D)
   * Compute L1 distance for rows of M between samples and apply negative exponent ![cb](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20c_b%20%28x_i%2C%20x_j%29%20%3D%20%5Cexp%28-%5CVert%20M_%7Bi%2Cb%7D%20-%20M_%7Bj%2Cb%7D%20%5CVert_%7BL_1%7D%29)
   * The output for this sample is the sum of the distance to all other samples: ![sum distance](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20o%28x_i%29_b%20%3D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%20c_b%28x_i%2C%20x_j%29%20%5Cin%20%5Cmathbb%7BR%7D)  
   ![sum](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20o%28x_i%29%20%3D%20%5Bo%28x_i%29_1%20%2C%20o%28x_i%29_2%20%2C%20...%20%2C%20o%28x_i%29_B%5D%20%5Cin%20%5Cmathbb%7BR%7D%20%5E%20B)  
   ![out](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20o%28X%29%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20B%7D)
   * The outputs O are concatenated to the features F and fed to the next layer of  the discriminator.
   <br><br>These minibatch features are calculated seperately for the training data and generated data.

3. **Historical Averaging:**

   * Adding a historical average of the parameters to each player's cost in an online fashion.
   * Shows to improve convergence and prevent the training procedure to fall into an orbit.

      ![historical averaging](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5CVert%20%5Ctheta%20-%20%5Cfrac%7B1%7D%7Bt%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bt%7D%20%5Ctheta%20%5Bi%5D%20%5CVert%5E2)

4. **One-sided Label Smoothing:**

   * Replacing the 1 and 0 in the discriminator's targets to alpha and beta.
   * This shows to increase robustness to adversarial examples.
   * The optimal discriminator becomes: ![optimal discriminator](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20D%28x%29%20%3D%20%5Cfrac%7B%5Calpha%20p_%7Bdata%7D%28x%29%20&plus;%20%5Cbeta%20p_%7Bmodel%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29%20&plus;p_%7Bmodel%7D%28x%29%7D)
   * We leave the negative label to zero because it does not help to improve training.

5. **Virtual Batch Normalization:**

   Similar to batch normalization but we normalize each sample on a reference batch of examples picked at the start of training. It is computationally expensive so we only use it on the generator newtork.


## Inception Score

We get the conditional distribution p(y|x) by applying the Inception model to each generated image from the GAN. If p(y|x) is low entropy, that means the images are providing useful information. We also expect the distribution of Images to have high entropy which means the network is generating varied images.

![inception score](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Cexp%28%5Cmathbb%7BE%7D_x%20KL%20%28p%28y%7Cx%29%20%5CVert%20p%28y%29%29%29)

This metric corresponds well to human judgement.

## Semi-Supervised Learning

For any standard classifier we can:

* Add samples from the generator to the dataset, labeled as y = K + 1 (Fake class).
* Half of the dataset comes from real data and the other half from generated data.
* The output of the discriminator becomes a K+1 dimensional vector.
* The loss for training the classifier becomes:

   ![loss](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20L%20%3D%20L_%7Bsupervised%7D%20&plus;%20L_%7Bunsupervised%7D)

   ![lsupervised](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20L_%7Bsupervised%7D%20%3D%20-%20%5Cmathbb%7BE%7D_%7Bx%2C%20y%20%5Csim%20p_%7Bdata%7D%28x%2Cy%29%7D%20%5Clog%28p_%7Bmodel%7D%28y%7Cx%2Cy%20%3C%20K%20&plus;%201%29%29)

   ![unsupervised](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Ctiny%20L_%7Bunsupervised%7D%20%3D%20-%5C%7B%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%28x%29%7D%20%5Clog%5B1%20-%20p_%7Bmodel%7D%28y%20%3D%20K%20&plus;%201%20%7C%20x%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20G%7D%20%5Clog%5Bp_%7Bmodel%7D%28y%3DK&plus;1%7Cx%29%5D%20%5C%7D)
   
Semi-supervised learning is shown to improve image quality and train a stronger classifier. This may be because that by training the discriminator to classify the images, we develop an internal representation that puts emphasize on features similar to the ones humans emphasize on.


## Results

Minibatch discrimination shows to generate better visual results and perform faster than feature matching. However feature matching is the better approach to training a stronger classifier.

Several networks were trained on the MNIST, CIFAR-10, SVHN and ImageNet datasets.
