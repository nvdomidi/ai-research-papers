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
   * Multiply it by tensor ![tensor](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B120%7D%20T%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BA%20%5Ctimes%20B%20%5Ctimes%20C%7D) to get 




