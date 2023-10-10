# Scaling of Class-wise Training Losses for Post-hoc Calibration(ICML 2023)

This repository is the official implementation code of our paper

## Installation
```
git clone https://github.com/SeungjinJung/SCTL.git
```

## Datasets

Please download from the below links and save to './datasets/'


Balanced Datasets : [Github](https://github.com/markus93/NN_calibration)

Long-tailed datasets : [Download](https://drive.google.com/drive/folders/1KfDriNxfnuqnmsj_zwpK3j7y6Lav7XBL?usp=share_link)

## Usage
```
python main.py --dataset cifar10_resnet110 --cal TS --loss CL --trainlog --name TS_CL_cifar10_resent110.log
```
## Citation
```

@InProceedings{pmlr-v202-jung23a,
  title = 	 {Scaling of Class-wise Training Losses for Post-hoc Calibration},
  author =       {Jung, Seungjin and Seo, Seungmo and Jeong, Yonghyun and Choi, Jongwon},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {15421--15434},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
}

```
