# Scaling of Class-wise Training Losses for Post-hoc Calibration
## Getting Started
### Installation
- Clone this repository :
```
git clone https://github.com/SeungjinJung/SCTL.git
```
- Download Datasets

  Pretrained model logits on balanced datasets : [Download](https://github.com/markus93/NN_calibration)
  
  Pretrained model logits on Long-tailed datasets : [Download](https://drive.google.com/drive/folders/1KfDriNxfnuqnmsj_zwpK3j7y6Lav7XBL?usp=share_link) (Direct Link)

### SCTL Train/Test
- Command the following code line 
```
python main.py --dataset cifar10_resnet110 --cal TS --loss CL --trainlog --name TS_CL_cifar10_resent110.log
```
## Citation
```
Not prepared
```
