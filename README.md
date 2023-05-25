# Scaling of Class-wise Training Losses for Post-hoc Calibration

This repogitory [Paper]

### Installation
```
git clone https://github.com/SeungjinJung/SCTL.git
```

### Datasets

Pretrained model logits on balanced datasets : [Download](https://github.com/markus93/NN_calibration) (Markus et al.)

Pretrained model logits on Long-tailed datasets : [Download](https://drive.google.com/drive/folders/1KfDriNxfnuqnmsj_zwpK3j7y6Lav7XBL?usp=share_link) (Direct Link)

### Usage
```
python main.py --dataset cifar10_resnet110 --cal TS --loss CL --trainlog --name TS_CL_cifar10_resent110.log
```
## Citation
```
Not prepared
```
