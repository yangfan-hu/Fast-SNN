# Fast-SNN
This repo holds the codes for Fast-SNN.

## Dependencies
* Python 3.8.8
* Pytorch 1.8.1

## Prepare Quantized ANNs
For training quantized ANNs, we follow the protocol defined in [Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks](https://openreview.net/group?id=ICLR.cc/2020/Conference)

For more details, please refer to [APoT_Quantization](https://github.com/yhhhli/APoT_Quantization)

## Image Classification


### CIFAR-10

#### Architectures
For network architectures, we currently support AlexNet, VGG11 (in 'CIFAR10'), ResNet-20/32/44/56/110 (in 'CIFAR-10'), and [ResNet-18](https://github.com/Gus-Lab/temporal_efficient_training) (in 'CIFAR10_resnet18'). For AlexNet, VGG11, and ResNet-20/32/44/56/110, we quantize both weights and activations. For ResNet-18, we quantize activations. 

#### Dataset
By default, the dataset is supposed to be in a 'data' folder at the same lavel of 'main.py'

#### Train Quantized ANNs
We progressively train full precision, 4, 3, and 2 bit ANN models.

An example to train AlexNet:
```
python main.py --arch alex --bit 32 --wd 5e-4
python main.py --arch alex --bit 4 --wd 1e-4  --lr 4e-2 --init result/alex_32bit/model_best.pth.tar
python main.py --arch alex --bit 3 --wd 1e-4  --lr 4e-2 --init result/alex_4bit/model_best.pth.tar
python main.py --arch alex --bit 2 --wd 3e-5  --lr 4e-2 --init result/alex_3bit/model_best.pth.tar
```

#### Evaluate Converted SNNs 
The time steps of SNNs are automatically calculated from activation precision, i.e., T = 2^b-1.
```
optinal arguments:
    --u                    Use unsigned IF neuron model
```
Example: AlexNet(SNN) performance with traditional unsigned IF neuron model. An 3/2-bit ANN is converted to an SNN with T=3/7.
```
python snn.py --arch alex --bit 3 -e -u --init result/alex_3bit/model_best.pth.tar
python snn.py --arch alex --bit 2 -e -u --init result/alex_2bit/model_best.pth.tar
```
Example: AlexNet(SNN) performance with signed IF neuron model. An 3/2-bit ANN is converted to an SNN with T=3/7.
```
python snn.py --arch alex --bit 3 -e -u --init result/alex_3bit/model_best.pth.tar
python snn.py --arch alex --bit 2 -e -u --init result/alex_2bit/model_best.pth.tar
```

#### Fine-tune Converted SNNs
By default, we use signed IF neuron model during fine-tuning. 

```
optinal arguments:
    --num_epochs / -n               Number of epochs to fine-tune at each layer
                                    default: 1
    --force                         Always update fine-tuned parameters without evaluation on training data
```

Example: finetune converted SNN models. 
```
python snn_ft.py --arch alex --bit 2 --force --init result/alex_2bit/model_best.pth.tar
python snn_ft.py --arch resnet18 --bit 2 --force --init result/resnet18_2bit/model_best.pth.tar
python snn_ft.py --arch resnet56 --bit 2 -n 8 --init result/resnet18_2bit/model_best.pth.tar
```

### ImageNet
We use distributed data parallel (DDP) for training. Please refer to Pytorch [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for details.

To speed up data loading, we replace the vanilla [Pytorch](https://pytorch.org/vision/0.8/datasets.html) dataloader with [nvidia-dali](https://developer.nvidia.com/dali).

Nvidia-dali package
```bash
# for CUDA 10
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
# for CUDA 11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

For more details on nvidia-dali, please refer to NVIDIA's official document [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)

#### Architectures
For network architectures, we currently support AlexNet and VGG16.

#### Train Qantized ANNs
With full-precision pre-trained models from [TorchVision](https://pytorch.org/vision/stable/index.html), we progressively 4, 3, and 2 bit ANN models.

An example to train AlexNet:
```
python -m torch.distributed.launch --nproc_per_node=4 dali_main.py -a alexnet -b 256 --bit 4 --workers 4 --lr=0.1 --epochs 60 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 dali_main.py -a alexnet -b 256 --bit 3 --init result/alexnet_4bit/model_best.pth.tar --workers 4 --lr=0.01 --epochs 60 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 dali_main.py -a alexnet -b 256 --bit 2 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --lr=0.01 --epochs 60 --dali_cpu /data/imagenet2012
```

#### Evaluate Converted SNNs

Example: AlexNet (SNN) performance with traditional unsigned IF neuron model. A 3/2-bit ANN is converted to an SNN with T=7/3.
```
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e -u --bit 3 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e -u --bit 2 --init result/alexnet_2bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
```
Example: AlexNEt (SNN) performance with signed IF neuron model. A 3/2-bit ANN is converted to an SNN with T=7/3.
```
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e --bit 3 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 snn.py -a alexnet -b 256 -e --bit 2 --init result/alexnet_2bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
```
### Finetune converted SNNs 
By default, we use signed IF neuron model in fine-tuning. 

Example:
```
python -m torch.distributed.launch --nproc_per_node=4 snn_ft.py -a alexnet -b 128 --bit 3 -n 8 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 snn_ft.py -a alexnet -b 128 --bit 2 -n 8 --init result/alexnet_2bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
```

## Object Detection
We use [yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch) as the framework for object detection.

### Preparation 
About required packages and datasets, please refer to [README](https://github.com/yjh0410/yolov2-yolov3_PyTorch/blob/master/README.md) in [yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch) for preparation. In the 'object detection' folder, we also prepare a merged README detailing everything. 

### PASCAL VOC 2007

```
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 32
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 4 --init CHECKPOINT_PATH
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 3 --init CHECKPOINT_PATH
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 2 --init CHECKPOINT_PATH
```

```
python eval.py -d voc --cuda -v yolov2_tiny --bit 4 --spike --init CHECKPOINT_PATH
python eval.py -d voc --cuda -v yolov2_tiny --bit 3 --spike --init CHECKPOINT_PATH
python eval.py -d voc --cuda -v yolov2_tiny --bit 2 --spike --init CHECKPOINT_PATH
```

### MS COCO 2017


## Semantic Segmentation
We use [vedaseg](https://github.com/Media-Smart/vedaseg), an open source semantic segmentation toolbox based on PyTorch, as the framework for semantic segmentation.

### Preparation 
About required packages and datasets, please refer to [README](https://github.com/Media-Smart/vedaseg/blob/master/README.md) in [vedaseg](https://github.com/Media-Smart/vedaseg) for preparation. In the 'semantic segmentation' folder, we also prepare a merged README detailing everything. 

### PASCAL VOC 2012


### MS COCO 20117




