# Self-Pruning Neural Network

This is a PyTorch implementation of a feed-forward neural network that learns to prune its own weights dynamically during training. 

Instead of training a large model and stripping it down later, I added learnable "gates" to standard linear layers. Applying an L1 penalty to these gates forces the network to identify and sever useless connections as it learns. The result is a highly sparse but accurate architecture.

## Architecture and Design

Building a self-pruning layer requires balancing sparsity with the network's underlying math. These are the core engineering decisions for this implementation:

### 1. Kaiming Initialization
Standard torch.nn.Linear layers use Kaiming uniform initialization to keep activation variances stable and prevent vanishing or exploding gradients. 

If the new gate scores are initialized at 0.0, the sigmoid function immediately sets them to 0.5. This instantly halves the magnitude of every weight in the network at step zero and breaks the Kaiming math. To solve this, I initialized the gate_scores with a mean of 3.0, yielding a sigmoid output of ~0.95. The network starts unpruned, maintains its optimal variance, and allows the loss function to organically decide which gates to close.

### 2. The L1 Penalty
Achieving true sparsity requires a mechanism that drives gate values exactly to zero, rather than just making them small. 

Using an L2 penalty (sum of squared values) means the gradient force gets weaker as the gate approaches zero, resulting in many tiny, active weights. An L1 penalty (sum of absolute values) applies a constant downward pressure regardless of the gate's magnitude. This constant force overpowers the classification loss for unimportant weights and drives them aggressively to zero.

### 3. Normalizing the Sparsity Loss
When calculating the sparsity penalty, I took the mean of the gate values across the network instead of the raw sum. 

Summing the gates of 1.7 million parameters creates a massive raw penalty score, which would require a microscopic lambda ($\lambda$) like `1e-5` to balance against the Cross-Entropy loss. Taking the mean keeps the penalty scale stable regardless of network depth or width. This makes the architecture modular, though it requires larger $\lambda$ values (10 to 100) to apply sufficient pressure.

## Results

I trained the network on CIFAR-10 with different $\lambda$ values to evaluate the accuracy vs. sparsity trade-off.

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| 0.0 | 51.72% | 0.00% |
| 10.0 | 54.37% | 78.01% |
| 50.0 | 54.54% | 92.60% |
| 100.0 | 54.65% | 96.10% |

*Note: Accuracy improved slightly as the network became sparser. The L1 penalty acted as a strong regularizer and prevented overfitting on the training data.*

### Gate Distribution

To verify the pruning, I plotted the distribution of the final gate values for the most aggressive run ($\lambda = 100$).

![Distribution of Final Gate Values](distribution.png)

Note: The Y-axis is scaled logarithmically. The spike at 0.0 represents over 1.6 million pruned weights, confirming the L1 penalty worked as intended. The secondary cluster near 1.0 represents the surviving 4% of weights the network retained to solve the classification task.
