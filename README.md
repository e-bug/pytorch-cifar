# Train CIFAR-10 and CIFAR-100 with PyTorch

Train deep networks with [PyTorch](http://pytorch.org/) on the CIFAR-10 and CIFAR-100 datasets.

## Usage
Train [ResNet56](https://arxiv.org/abs/1512.03385) on CIFAR-10 
- with batch size `128`
- for`182` epochs
- with initial learning rate `0.1`
- and a piecewise constant learning rate decay function 
- with a decay factor of `0.1` (default) at epochs `91` and `136`
- using the first two GPUs
- storing `10` state checkpoints
- and printing a progress bar

```bash
# setup options
MODEL=resnet56
BATCH_SIZE=128
NUM_EPOCHS=182
NUM_CKPTS=10
LR=0.1
DECAY_POLICY=pconst
LR_MILESTONES="91 136"
export CUDA_VISIBLE_DEVICES=0,1

# run 
SCRIPT=main.py

python $SCRIPT \
  --model ${MODEL} \
  --batch_size ${BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --num_ckpts ${NUM_CKPTS} \
  --progress_bar \
  --lr ${LR} \
  --lr_decay_policy ${DECAY_POLICY} \
  --lr_milestones ${LR_MILESTONES}
```
```bash
==> Preparing data..
Files already downloaded and verified
Files already downloaded and verified
==> Building resnet56 model..

Epoch: 0
 [==>........................... 19/391 ..............................]  Step: 1s392ms | Tot: 24s295ms | lr: 1.000e-01 | Loss: 2.173 | Acc: 17.393% (423/2432)
```

## Accuracy (as reported by @kuangliu)
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

