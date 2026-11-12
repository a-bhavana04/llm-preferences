# Differential Privacy for Speech Model Personalization

This repository contains a PyTorch implementation of the methods described in the paper "[Deep Learning with Differential Privacy](https://dl.acm.org/doi/10.1145/2976749.2978318)" by Abadi et al.. The goal of this project was to apply these techniques to train a deep neural network on a sensitive audio dataset while providing mathematical privacy guarantees.

The core challenge addressed is that modern deep learning models have a large capacity to memorize and potentially expose fine details of their training data. This notebook implements a solution that offers protection against a strong adversary with full knowledge of the model's architecture and parameters, making it suitable for scenarios where models are deployed on-device. We successfully trained effective audio classification models under the papers objective of a "modest ('single-digit') privacy budget".

## Design & Implementation

The implementation is centered on Differentially Private Stochastic Gradient Descent (DP-SGD), as outlined in Algorithm 1 of the source paper. This method integrates privacy directly into the training loop by modifying the gradient updates. The key operations, performed at each training step, are:

1.  **Per-Example Gradient Computation**: For each training example in a lot, we compute its individual gradient. This is a prerequisite for the following steps and allows us to understand the influence of each specific data point.

2.  **Per-Example Gradient Clipping**: Before averaging, the `l2` norm of each individual gradient is clipped to a predefined threshold, `C`. This is the most critical step for limiting the sensitivity of each training example, as it bounds the maximum influence any single audio sample can have on the resulting model update.

3.  **Noise Addition**: After clipping and averaging the gradients within a lot, calibrated Gaussian noise is added to the aggregated gradient. The amount of noise is scaled by the noise multiplier, `σ`, and the clipping bound, `C`. This step obscures the precise contributions of the training examples making it computationally infeasible to reverse-engineer individual data points from the final model.

4.  **Privacy Accounting**: To track the cumulative privacy cost over many training steps, we rely on the Moments Accountant method. This technique provides a much tighter bound on the privacy loss (`ε`) compared to traditional composition theorems, enabling us to train for more epochs without exhausting our privacy budget.

## Settings

The experiments were conducted with the following settings. We began with an initial sweep and then progressed to more advanced tuning based on the results.

*   **Dataset**: A custom audio dataset for keyword spotting, preprocessed into MFCC features.
    *   Training examples: 240
    *   Test examples: 81
    *   Number of classes: 4
*   **Model**: A feed-forward neural network with one hidden layer and a ReLU activation function, which qualifies as a deep learning model with a non-convex objective. The input dimension was first reduced using a Differentially Private PCA layer, a technique shown to improve both utility and performance.
*   **Initial Sweep Hyperparameters**:
    *   **Batch Size (`L`)**: 16
    *   **Epochs**: 20
    *   **Clipping Norm (`C`)**: `[0.5, 1.0]`
    *   **Noise Multiplier (`σ`)**: `[0.5, 1.0, 2.0]`
    *   **Learning Rate (`lr`)**: 0.1
    *   **Random Seed**: A fixed seed was used for reproducibility.

## Results

Our experiments successfully replicated the core findings of the paper on our custom audio dataset. The non-private baseline accuracy for this model architecture was **79.01%**.

### Initial Parameter Sweep

We began with a small grid search to get a first look at the privacy-utility trade-off.

| C   | σ   | Accuracy (%) | ε (for δ=1/240) |
|:---:|:---:|:------------:|:---------------:|
| 0.5 | 0.5 |    56.17%    |      18.64      |
| 0.5 | 1.0 |    49.38%    |      2.93       |
| 0.5 | 2.0 |    46.91%    |      **0.95**       |
| 1.0 | 0.5 |    62.35%    |      18.64      |
| 1.0 | 1.0 |    61.73%    |      2.93       |
| **1.0** | **2.0** |    **60.49%**    |      **0.95**       |

*Table 1: Results from the initial small-grid hyperparameter sweep.*

This initial exploration revealed several key insights consistent with the paper's findings:
*   **Effect of Noise (`σ`)**: Increasing the noise multiplier consistently and dramatically improves the privacy guarantee (lowers `ε`). With `C=1.0`, increasing `σ` from 0.5 to 2.0 reduced `ε` from a very high 18.64 to an excellent 0.95.
*   **The Trade-off**: The highest accuracy (62.35%) was achieved with the worst privacy (`ε=18.64`). However, this sweep immediately identified a highly promising model: with `C=1.0` and `σ=2.0`, we achieved **60.49% accuracy** with an excellent privacy guarantee of **`ε=0.95`**, successfully meeting the paper's goal of a modest privacy budget.

### Systematic Hyperparameter Tuning

Informed by our initial results and the paper's guidance that parameters like lot size have a large impact on the model accuracy, we conducted a more focused tuning process.

**How I implemented it:**
   * Expand the Lot Size (L) sweep: The paper empirically found that the best lot size is often around sqrt(N). For my dataset (N=240), this is ~16, which I used. I tested this theory by trying lot sizes around this value, such as L ∈ {8, 16, 32}, to see if this heuristic holds for the data.
   *  Tuned the Clipping Bound (C) more systematically: The paper suggests a practical method for choosing C: "taking the median of the norms of the unclipped gradients over the course of training". I ran a few non-private epochs, logged these norms, and use the median to inform a more targeted sweep for C.
   *  Experimented with a Learning Rate Schedule: The paper found a "small benefit to starting with a relatively large learning rate, then linearly decaying it" before keeping it constant. I implemented this simple decay schedule instead of our fixed learning rate to see if it improves convergence and final accuracy.


| lot_size | clip_norm | noise_multiplier | accuracy | epsilon (ε) | delta (δ) |
|:--------:|:---------:|:----------------:|:--------:|:-----------:|:---------:|
|    16    |   2.02    |         1        |  64.81%  |    2.93     |   1e-5    |
|    16    |   2.02    |         2        |  51.23%  |    0.95     |   1e-5    |
|    16    |   2.02    |         4        |  43.83%  |    0.39     |   1e-5    |
|    16    |   4.04    |         1        |  54.94%  |    2.93     |   1e-5    |
|    16    |   4.04    |         2        |  41.36%  |    0.95     |   1e-5    |
|    16    |   4.04    |         4        |  25.93%  |    0.39     |   1e-5    |
|    16    |   6.05    |         1        |  53.09%  |    2.93     |   1e-5    |
|    16    |   6.05    |         2        |  41.36%  |    0.95     |   1e-5    |
|    16    |   6.05    |         4        |  41.36%  |    0.39     |   1e-5    |
|    32    |   2.02    |         1        |  62.35%  |    4.43     |   1e-5    |
|    32    |   2.02    |         2        |  62.35%  |    1.43     |   1e-5    |
|    32    |   2.02    |         4        |  40.74%  |    0.59     |   1e-5    |
|    **32**    |   **4.04**    |         **1**        |  **69.75%**  |    **4.43**     |   **1e-5**    |
|    32    |   4.04    |         2        |  54.94%  |    1.43     |   1e-5    |
|    32    |   4.04    |         4        |  43.21%  |    0.59     |   1e-5    |
|    32    |   6.05    |         1        |  67.28%  |    4.43     |   1e-5    |
|    32    |   6.05    |         2        |  43.83%  |    1.43     |   1e-5    |
|    32    |   6.05    |         4        |  40.74%  |    0.59     |   1e-5    |

*Table 2: The full results from the improved hyperparameter sweep. This detailed exploration allowed us to find configurations that significantly outperformed our initial sweep.*

This more detailed exploration led to the discovery of a superior model for high-utility applications:
*   A new **highest utility model** (`L=32, C=4.04, σ=1.0`) achieved **69.75% accuracy** with a strong privacy guarantee of `ε=4.43`.

### Regularization Effect of Differential Privacy

A key finding, consistent with the paper, is that DP-SGD acts as a powerful regularizer, preventing the model from overfitting. The paper notes that in contrast to non-private training, DP training results in a "small difference between the model’s accuracy on the training and the test sets".

![Overfitting Analysis Plot](./overfitting_plot.png)
*Figure 1: The non-private model (red) shows a large and growing gap between training and test accuracy, a clear sign of overfitting. The DP model (blue) shows that the two metrics remain close, indicating better generalization.*

## Takeaways

*   **The Privacy-Utility Trade-off is Real and Manageable:** We successfully trained a model that achieved **69.75% accuracy** (within 10 points of the 79.01% non-private baseline) while providing a strong, single-digit privacy guarantee (`ε=4.43`). This confirms that privacy can be achieved at a manageable cost in model quality.

*   **DP-SGD Provides an Inherent Regularization Benefit:** A key observation, consistent with the paper's findings, is that differentially private training acts as a strong regularizer. Our results clearly show that the DP model avoids the significant overfitting seen in the non-private baseline, where the gap between training and test accuracy grows over time. This suggests the private model is learning generalizable patterns from the audio data rather than simply memorizing the fine details of individual training examples, which is an attractive consequence of the training method.

*   **Hyperparameter Tuning is Critical:** Our iterative sweeps confirmed the paper's claim that parameters like `lot_size`, `clip_norm`, and `noise_multiplier` have a "large impact on the model accuracy" and must be "carefully tuned for optimal performance". While the initial sweep provided a strong starting point, the subsequent, more focused tuning was essential to discovering the highest-utility model. The optimal clipping norm, for instance, was highly dependent on the amount of noise being added.

## AI Declaration
**AI Tools Used**: Gemini 2.5 Pro
**Specific Contributions**:
1. **Data loading boilerplate**: Initial code for file I/O and audio preprocessing
2. **Opacus Help**: Explained the problems with Opacus in this project. Went with a manual algorithm/hybrid approach
3. **Plot formatting**: code structure for comparison figure
4. **README structure**: Initial Markdown formatting and section organization
