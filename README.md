# Self-Pruning Neural Network

A PyTorch implementation of a feed-forward neural network that dynamically learns to prune its own weights during training. 

Instead of applying post-training pruning, this network augments standard linear layers with learnable "gate" parameters. By applying an L1 regularization penalty to these gates, the network is forced to identify and sever its own weakest connections on the fly, leaving behind a highly sparse but accurate architecture.

## Architectural Decisions

Building a self-pruning layer from scratch requires balancing the sparsity penalty with the mathematical stability of the network. Here are the core engineering decisions behind this implementation:

### 1. Preserving Kaiming Initialization
Standard `torch.nn.Linear` layers use Kaiming uniform initialization to keep activation variances stable and prevent vanishing/exploding gradients. 

If we initialize the gate scores at `0.0`, the sigmoid function immediately converts them to `0.5`. This effectively halves the magnitude of every weight in the network before training even begins, destroying the Kaiming variance guarantees. To fix this, the `gate_scores` are initialized with `mean=3.0` (yielding a sigmoid output of ~0.95). The network starts almost completely unpruned, utilizing the optimal weight variance, and then relies on the loss function to learn which gates to close.

### 2. The L1 Penalty Intuition
To achieve true sparsity, the network needs a mechanism that drives gate values exactly to zero, rather than just making them small. 

If we used an L2 penalty (sum of squared values), the gradient force would decrease as the gate approached zero, resulting in many small but active weights. An L1 penalty (sum of absolute values) applies a *constant* downward gradient pressure regardless of the gate's magnitude. This constant pressure easily overpowers the Cross-Entropy classification loss for unimportant weights, driving them aggressively and exactly to zero.

### 3. Sparsity Loss Normalization
When calculating the sparsity penalty, this implementation takes the **mean** of the gate values across the network rather than the raw **sum**. 

Summing the gates of ~1.7 million parameters would result in a massive raw penalty score, requiring a microscopic $\lambda$ (e.g., `1e-5`) to balance against a standard Cross-Entropy loss. By taking the mean, the sparsity penalty scale remains perfectly stable regardless of how wide or deep the network gets. This makes the architecture highly modular, though it means our optimal $\lambda$ values sit in the `10 - 100` range to exert enough pressure.

## Results

The network was trained on CIFAR-10 across multiple $\lambda$ values to demonstrate the trade-off between classification accuracy and network sparsity. 

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| 0.0 | 51.72% | 0.00% |
| 10.0 | 54.37% | 78.01% |
| 50.0 | 54.54% | 92.60% |
| 100.0 | 54.65% | 96.10% |

*Note: Interestingly, accuracy slightly improved alongside sparsity in this specific run, suggesting the L1 penalty effectively acted as a strong regularizer against overfitting on the training data.*

### Gate Distribution

To verify the pruning mechanism, we can look at the distribution of the final gate values for the most aggressive model ($\lambda = 100$).

![Distribution of Final Gate Values](distribution.png)

*Note: The Y-axis is scaled logarithmically.* The massive spike at `0.0` (representing over 1.6 million pruned weights) proves the L1 penalty functioned exactly as intended. The visible secondary cluster building toward `1.0` represents the surviving ~4% of weights the network deemed absolutely critical to solving the classification task.
