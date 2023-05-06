# Development of Domain Agnostic Contrastive Learning
## Intuition
Learn a distribution of the dataset, then using mixup to construct augmented image views.
## Known issues
The contrastive loss is not synced among distributed GPUs.
