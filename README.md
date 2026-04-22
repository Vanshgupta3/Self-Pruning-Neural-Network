# Self-Pruning-Neural-Network
Deploying large neural networks in production is often constrained by memory footprints and inference 
latency budgets. Pruning — the removal of redundant weights — is a well-established technique for 
compressing neural networks. This case study implements a more ambitious variant: a network that learns 
to prune itself during training, without any post-training surgery. 
The central idea is to attach a learnable scalar gate to every weight. A sigmoid function squashes each 
gate into the interval (0, 1). An L1 penalty in the loss function continuously pressures gates toward zero, 
effectively removing the weights they control. Weights that genuinely contribute to reducing 
classification loss resist this pressure; those that do not collapse to zero and are pruned. 
