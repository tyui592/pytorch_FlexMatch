FlexMatch
==

**Unofficial Pytorch Implementation of "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"**

## Usage
### Requirements
* wandb (Optional)
* pytorch (v2.0.1)
* torchvision (v0.15.2)
* PIL (v9.4.0)

### Arguments
* Augmentation Policy (`--augs`)
  - 0: no augmentation
  - 1: weak augmentation
  - 2: strong augmentation (based on RandAug)
* Check [`config.py`](./config.py) file for details. (Default parameters are set for cifar10)

### Example Scripts
```bash
# Model Training
$ python main.py --mode 'train' --data 'cifar10' --num_X 4000 --augs 1 2  --nesterov --amp --include_x_in_u --save_path ./model-store/001
>>>...
>>>Sun Aug  6 15:18:28 2023: Iteration: [1044480/1048576], Ls: 0.1156, Lu: 0.1153, Mask: 0.9928, Acc(train/test): [1.0000/0.9580]
>>>Sun Aug  6 15:25:34 2023: Iteration: [1045504/1048576], Ls: 0.1158, Lu: 0.1154, Mask: 0.9932, Acc(train/test): [0.9999/0.9582]
>>>Sun Aug  6 15:32:40 2023: Iteration: [1046528/1048576], Ls: 0.1146, Lu: 0.1143, Mask: 0.9932, Acc(train/test): [1.0000/0.9583]
>>>Sun Aug  6 15:39:45 2023: Iteration: [1047552/1048576], Ls: 0.1138, Lu: 0.1135, Mask: 0.9933, Acc(train/test): [1.0000/0.9585]
>>>Sun Aug  6 15:46:50 2023: Iteration: [1048576/1048576], Ls: 0.1147, Lu: 0.1143, Mask: 0.9934, Acc(train/test): [1.0000/0.9580]

# Model Evaluation
$ python main.py --mode 'eval' --load_path ./model-store/001/ckpt.pth
>>>...
>>>Model Performance: 0.9580
```

## Results

### CIFAR10
| Num Labaled Data | Top 1 Acc |
| --- | --- | 
| 4000 | 0.9580 | 
| 250 | 0.9505 |
| 40 | on going |
**Model weights (and training logs) of the above performance are on [the release page](https://github.com/tyui592/pytorch_FlexMatch/releases/tag/v0.1).**

## References
- https://arxiv.org/abs/2110.08263
- https://www.zijianhu.com/post/pytorch/ema/
- https://github.com/kekmodel/FixMatch-pytorch
