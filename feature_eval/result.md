# Linear eval with resnet18 as the backbone
## Pretrain
* epochs = 100
* batch_size = 128 * 4
* lr = 0.3 * batch_size / 256
* w = 1e-6
* optimizer = LARS
* warmup = 10
* CosineAnnealingLRScheduler

## Linear eval
* epochs = 100
* batch_size = 1024
* lr = 0.1 * batch_size / 256
* w = 0
* optimizer = LARS
## Result
77.45 Top-1 Accuracy