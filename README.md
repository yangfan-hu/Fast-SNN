# Fast-SNN
This repo holds the codes for .

## Dependencies
* Python 3.8.8
* Pytorch 1.8.1

## Prepare Quantized ANNs
For training quantized ANNs, we follow the protocol defined in [Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks](https://openreview.net/group?id=ICLR.cc/2020/Conference)

For their detailed codes, please refer to [APoT_Quantization](https://github.com/yhhhli/APoT_Quantization)

## CIFAR-10

We progressively train full precision, 4, 3, and 2 bit ANN models.
```
python main.py --arch alex --bit 32 -id 2 --wd 5e-4
python main.py --arch alex --bit 4 -id 2 --wd 1e-4  --lr 4e-2 --init result/alex_32bit/model_best.pth.tar
python main.py --arch alex --bit 3 -id 2 --wd 1e-4  --lr 4e-2 --init result/alex_4bit/model_best.pth.tar
python main.py --arch alex --bit 2 -id 2 --wd 3e-5  --lr 4e-2 --init result/alex_3bit/model_best.pth.tar
```

Evaluate SNN performance with traditional unsigned IF neuron model. An 3/2-bit ANN is converted to an SNN with T=3/7.
```
python snn.py --arch alex --bit 3 -id 2 -e -u --init result/alex_3bit/model_best.pth.tar
python snn.py --arch alex --bit 2 -id 2 -e -u --init result/alex_2bit/model_best.pth.tar
```
Evaluate SNN performance with signed IF neuron model. An 3/2-bit ANN is converted to an SNN with T=3/7.
```
python snn.py --arch alex --bit 3 -id 2 -e -u --init result/alex_3bit/model_best.pth.tar
python snn.py --arch alex --bit 2 -id 2 -e -u --init result/alex_2bit/model_best.pth.tar
```
Finetune converted SNN models. By default, we use signed IF neuron model in fine-tuning. 
```
python snn_ft.py --arch alex --bit 3 -id 2  -n 8 --init result/alex_3bit/model_best.pth.tar
python snn_ft.py --arch alex --bit 2 -id 2  -n 8 --init result/alex_2bit/model_best.pth.tar
```

## ImageNet

With 32-bit pre-trained models from torchvision, we progressively 4, 3, and 2 bit ANN models.
```
python -m torch.distributed.launch --nproc_per_node=4 dali_main.py -a alexnet -b 256 --bit 4 --workers 4 --lr=0.025 --epochs 60 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 dali_main.py -a alexnet -b 256 --bit 3 --init result/alexnet_4bit/model_best.pth.tar --workers 4 --lr=0.0025 --epochs 60 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 dali_main.py -a alexnet -b 256 --bit 2 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --lr=0.0025 --epochs 60 --dali_cpu /data/imagenet2012
```
Evaluate SNN performance with traditional unsigned IF neuron model. A 3/2-bit ANN is converted to an SNN with T=3/7.
```
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e -u --bit 3 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e -u --bit 2 --init result/alexnet_2bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
```
Evaluate SNN performance with signed IF neuron model. A 3/2-bit ANN is converted to an SNN with T=3/7.
```
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e --bit 3 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e --bit 2 --init result/alexnet_2bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
```
Finetune converted SNN models. By default, we use signed IF neuron model in fine-tuning. 
```
python -m torch.distributed.launch --nproc_per_node=4 snn_ft.py -a alexnet -b 128 --bit 3 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 snn_ft.py -a alexnet -b 128 --bit 2 --init result/alexnet_2bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
```
