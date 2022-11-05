# Fast-SNN
This repo holds the codes for .

## Dependencies
* Python 3.8.8
* Pytorch 1.8.1

## Prepare Quantized ANNs
For training quantized ANNs, we follow the protocol defined in [Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks](https://openreview.net/group?id=ICLR.cc/2020/Conference)

For more code details, please refer to [APoT_Quantization](https://github.com/yhhhli/APoT_Quantization)

## CIFAR-10

```
python main.py --arch alex --bit 32 -id 2 --wd 5e-4
python main.py --arch alex --bit 4 -id 2 --wd 1e-4  --lr 4e-2 --init result/alex_32bit/model_best.pth.tar
python main.py --arch alex --bit 3 -id 2 --wd 1e-4  --lr 4e-2 --init result/alex_4bit/model_best.pth.tar
python main.py --arch alex --bit 2 -id 2 --wd 3e-5  --lr 4e-2 --init result/alex_3bit/model_best.pth.tar
```




## ImageNet
