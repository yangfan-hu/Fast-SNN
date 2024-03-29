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
By default, we use signed IF neuron model.
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
python snn_ft.py --arch resnet56 --bit 2 -n 8 --init result/resnet56_2bit/model_best.pth.tar
```
#### Checkpoints for Quantized Models

|   Model   |   3-bit   |   2-bit  |
| :-------: | --------- | -------- |  
| AlexNet  | [alex_3bit](https://drive.google.com/file/d/1V2sd5qgDsrrM0gpleiSe61vzBUbPTOZu/view?usp=drive_link)  | [alex_2bit](https://drive.google.com/file/d/1_EoFWpTNKX7eGmSJMbvMnzAF1bEGFaKQ/view?usp=drive_link) | 
| VGG11    | [vgg11_3bit](https://drive.google.com/file/d/1KCagNVOENktLXNAfQlj4ih74zcmckJU6/view?usp=drive_link)  | [vgg11_2bit](https://drive.google.com/file/d/1ogi_ScbHFSnEZaJAaAz7aGvo1FkXOUNy/view?usp=drive_link) | 
| ResNet20 | [resnet20_3bit](https://drive.google.com/file/d/1k9LqfJz7EXG_jT3X9PVR3Rbk00tB6Lsm/view?usp=drive_link)  | [resnet20_2bit](https://drive.google.com/file/d/13AFv8B1JDKDjy4sspyzFsSVWNOb9oA8v/view?usp=drive_link) | 
| ResNet44 | [resnet44_3bit](https://drive.google.com/file/d/1htWdb7mNN2_Id3QaiGO0_eG8OVXXwGH8/view?usp=drive_link)  | [resnet44_2bit](https://drive.google.com/file/d/1VNOjCbTZV-t0GC-b-7_GNOHSqmxTSzqN/view?usp=drive_link) | 
| ResNet56 | [resnet56_3bit](https://drive.google.com/file/d/1i6cJU4S7pXkA60JBRHUOkklGrNVyBiSX/view?usp=drive_link)  | [resnet56_2bit](https://drive.google.com/file/d/1Xr87673Ur51snCoLtUSedQM-prOXichC/view?usp=drive_link) | 
| ResNet18 | [resnet18_3bit](https://drive.google.com/file/d/1t0npKS-Mt-OQYXL7mY3G1yXp6v40mAg0/view?usp=drive_link)  | [resnet18_2bit](https://drive.google.com/file/d/1XIFKSyI3LOZ9hWykQ7N3I-tISphVqWYF/view?usp=drive_link) | 

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
#### Finetune converted SNNs 
By default, we use signed IF neuron model in fine-tuning. 

Example:
```
python -m torch.distributed.launch --nproc_per_node=4 snn_ft.py -a alexnet -b 128 --bit 3 -n 8 --init result/alexnet_3bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
python -m torch.distributed.launch --nproc_per_node=4 snn_ft.py -a alexnet -b 128 --bit 2 -n 8 --init result/alexnet_2bit/model_best.pth.tar --workers 4 --dali_cpu /data/imagenet2012
```
#### Checkpoints for Quantized Models

|   Model   |   3-bit   |   2-bit  |
| :-------: | --------- | -------- |  
| AlexNet   | [alexnet_3bit](https://drive.google.com/file/d/1c-UXESDKheXWkwNxmIjQ1WdBwO5AkwU1/view?usp=drive_link)  | [alexnet_2bit](https://drive.google.com/file/d/1HfNpwjAi91j48Zy1nWtu4sReQ-TZaUHG/view?usp=drive_link)  | 
| VGG16     | [vgg16_3bit](https://drive.google.com/file/d/1oZOATIzNfr4NzPVOT8OB3rT6Exlh8cXa/view?usp=drive_link)  | [vgg16_2bit](https://drive.google.com/file/d/1mqpg0iG7mHVOlRT7R-zlkYRzZzAPRWBv/view?usp=drive_link) | 

## Object Detection
We use [yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch) as the framework for object detection.

### Preparation 
About required packages and datasets, please refer to [README](https://github.com/yjh0410/yolov2-yolov3_PyTorch/blob/master/README.md) in [yolov2-yolov3_PyTorch](https://github.com/yjh0410/yolov2-yolov3_PyTorch) for preparation. In the 'object detection' folder, we also prepare a merged [README](https://github.com/yangfan-hu/Fast-SNN/blob/main/object%20detection/README.md) detailing everything. 

### Architecture
We currently support Tiny YOLO and YOLOv2 with a ResNet-34 backbone. 

```
optinal arguments:
    --version / -v               Supported architecture
                                 available: yolov2_tiny, yolov2_r34
```


### PASCAL VOC 2007


#### Train Quantized ANNs

Example: train Tiny YOLO with activations qunatized to 32/4/3/2 bits. 
```
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 32
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 4 --init CHECKPOINT_PATH
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 3 --init CHECKPOINT_PATH
python -m torch.distributed.launch --nproc_per_node=4 train.py -d voc -v yolov2_tiny -ms --ema --sybn --batch_size 4 --bit 2 --init CHECKPOINT_PATH
```
#### Evaluate Models

```
optinal arguments:
    --spike               Evaluate with spikes (as SNNs)
```

Example: evaluate Tiny YOLO (SNN) with T = 15, 7, 3
```
python eval.py -d voc --cuda -v yolov2_tiny --bit 4 --spike --init CHECKPOINT_PATH
python eval.py -d voc --cuda -v yolov2_tiny --bit 3 --spike --init CHECKPOINT_PATH
python eval.py -d voc --cuda -v yolov2_tiny --bit 2 --spike --init CHECKPOINT_PATH
```

#### Checkpoints for Quantized Models

|   Model           |   4-bit   |   3-bit  |   2-bit  |
| :-------:         | --------- | -------- | -------- | 
| Tiny Yolo         | [yolov2_tiny_4bit](https://drive.google.com/file/d/1ffQyGvQSV-yOytGxK9JkWC1bUsq7uxbU/view?usp=drive_link) | [yolov2_tiny_3bit](https://drive.google.com/file/d/1GKJqxAM1k470Xfw7jBKDpff6SnzsayS3/view?usp=drive_link) | [yolov2_tiny_2bit](https://drive.google.com/file/d/1IvrYjyxnbqiYhfHEDzERU7AjV5hb3_pI/view?usp=drive_link) | 
| YoloV2(ResNet-34) | [yolov2_r34_4bit](https://drive.google.com/file/d/1qfI4wM7rxY_VlQBrdcIaS6ylTgK3xOhZ/view?usp=drive_link)  | [yolov2_r34_3bit](https://drive.google.com/file/d/1bZvrol-vf9iSDQdc-uE8V2wB3W4t39G2/view?usp=drive_link)  | [yolov2_r34_2bit](https://drive.google.com/file/d/1yJ5H5DALv8f9uKYyYpwGT8ypfsVzAZ0S/view?usp=drive_link)  | 

### MS COCO 2017

##### Train Quantized ANNs

Example: train Tiny YOLO with activations qunatized to 32/4/3/2 bits. 
```
python -m torch.distributed.launch --nproc_per_node=4 train.py -d coco -v yolov2_tiny --bit 32 -ms --ema --sybn --batch_size 4 
python -m torch.distributed.launch --nproc_per_node=4 train.py -d coco -v yolov2_tiny --bit 4 -ms --ema --sybn --batch_size 4  --init CHECKPOINT_PATH
python -m torch.distributed.launch --nproc_per_node=4 train.py -d coco -v yolov2_tiny --bit 3 -ms --ema --sybn --batch_size 4 --init CHECKPOINT_PATH
python -m torch.distributed.launch --nproc_per_node=4 train.py -d coco -v yolov2_tiny --bit 2 -ms --ema --sybn --batch_size 4 --init CHECKPOINT_PATH
```

##### Evaluate Models

Example: evaluate Tiny YOLO (SNN) with T = 15, 7, 3
```
python eval.py -d coco-val --cuda -v yolov2_tiny --bit 4 --spike --init CHECKPOINT_PATH
python eval.py -d coco-val --cuda -v yolov2_tiny --bit 3 --spike --init CHECKPOINT_PATH
python eval.py -d coco-val --cuda -v yolov2_tiny --bit 2 --spike --init CHECKPOINT_PATH 
```
#### Checkpoints for Quantized Models

|   Model           |   4-bit   |   3-bit  |   2-bit  |
| :-------:         | --------- | -------- | -------- | 
| Tiny Yolo         | [yolov2_tiny_4bit](https://drive.google.com/file/d/1bOqX8HiGTV18ZT_VX6eCEKXAVqLOZ8qC/view?usp=drive_link) | [yolov2_tiny_3bit](https://drive.google.com/file/d/1yPnTvt3OJNb_SsTXAeu5XCdwjwOxPOi7/view?usp=drive_link) | [yolov2_tiny_2bit](https://drive.google.com/file/d/147OgIpON1myK9EN-re4emrSWD_wCLf_F/view?usp=drive_link) | 
| YoloV2(ResNet-34) | [yolov2_r34_4bit](https://drive.google.com/file/d/1Z0v5J10tIMeaoOz9AvdETxcZk88FG4l4/view?usp=drive_link)  | [yolov2_r34_3bit](https://drive.google.com/file/d/1Xt7-GE1aIUhBmVNJ82G8sIEBVTfy6DIw/view?usp=drive_link) | [yolov2_r34_2bit](https://drive.google.com/file/d/1VlsSqqRnfzxL5ddls0_pMDVSTUdSiFbz/view?usp=drive_link) | 


## Semantic Segmentation
We use [vedaseg](https://github.com/Media-Smart/vedaseg), an open source semantic segmentation toolbox based on PyTorch, as the framework for semantic segmentation.

### Preparation 
About required packages and datasets, please refer to [README](https://github.com/Media-Smart/vedaseg/blob/master/README.md) in [vedaseg](https://github.com/Media-Smart/vedaseg) for preparation. In the 'semantic segmentation' folder, we also prepare a merged [README](https://github.com/yangfan-hu/Fast-SNN/blob/main/semantic%20segmentation/README.md) detailing everything. 

### Architecture 
We currently support Deeplabv1 (VGG9) and Deeplabv3 (ResNet-34 + ASPP).

### PASCAL VOC 2012

#### Train Quantized ANNs

Example: train VGG9 with activations qunatized to 32/4/3/2 bits. 
```
bash ./tools/dist_train.sh configs/voc_deeplabv1.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/voc_deeplabv1_4bit.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/voc_deeplabv1_3bit.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/voc_deeplabv1_2bit.py "0, 1, 2, 3" 
```

Example: train ResNet-34 + ASPP with activations qunatized to 32/4/3/2 bits. 
```
bash ./tools/dist_train.sh configs/voc_deeplabv3.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/voc_deeplabv3_4bit.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/voc_deeplabv3_3bit.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/voc_deeplabv3_2bit.py "0, 1, 2, 3" 
```
#### Evaluate Models

Example: evaluate VGG9 (SNN) with T = 15, 7, 3
```
bash ./tools/dist_test.sh configs/voc_deeplabv1_T15.py './workdir/voc_deeplabv1_4bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/voc_deeplabv1_T7.py './workdir/voc_deeplabv1_3bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/voc_deeplabv1_T3.py './workdir/voc_deeplabv1_2bit/best_mIoU.pth' "0, 1, 2, 3" 
```

Example: evaluate ResNet-34 + ASPP (SNN) with T = 15, 7, 3
```
bash ./tools/dist_test.sh configs/voc_deeplabv3_T15.py './workdir/voc_deeplabv3_4bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/voc_deeplabv3_T7.py './workdir/voc_deeplabv3_3bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/voc_deeplabv3_T3.py './workdir/voc_deeplabv3_2bit/best_mIoU.pth' "0, 1, 2, 3" 
```
#### Checkpoints for Quantized Models

|   Model           |   4-bit   |   3-bit  |   2-bit  |
| :-------:         | --------- | -------- | -------- | 
|     VGG-9         | [voc_deeplabv1_4bit](https://drive.google.com/file/d/1eBOI2JeMJeOQRX7LdD_Awv-jq8B-uava/view?usp=drive_link)  | [voc_deeplabv1_3bit](https://drive.google.com/file/d/1H_AHO1v6YFzYNNYGRDY7z8U6c70emA1W/view?usp=drive_link)  | [voc_deeplabv1_2bit](https://drive.google.com/file/d/12yp-Em3_nYXKQ9_5rLCZtuuXWlxbLndx/view?usp=drive_link)  | 
| ResNet-34 + ASPP  | [voc_deeplabv3_4bit](https://drive.google.com/file/d/1Q1jNy3Pbhv_4VZdEYkdl_fdUCjC3IuVl/view?usp=drive_link)  | [voc_deeplabv3_3bit](https://drive.google.com/file/d/1JBh4tweiHhV5-KL9fBTikirYqeyoIsdj/view?usp=drive_link) | [voc_deeplabv3_2bit](https://drive.google.com/file/d/1XLafo79ts4kM1i570X49L_BxEkpeALJd/view?usp=drive_link) | 


### MS COCO 2017

#### Train Quantized ANNs

Example: train VGG9 with activations qunatized to 32/4/3/2 bits. 
```
bash ./tools/dist_train.sh configs/coco_deeplabv1.py "0, 1, 2, 3, 6, 7" 
bash ./tools/dist_train.sh configs/coco_deeplabv1_4bit.py "0, 1, 2, 3, 6, 7" 
bash ./tools/dist_train.sh configs/coco_deeplabv1_3bit.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/coco_deeplabv1_2bit.py "0, 1, 2, 3" 
```
Example: train ResNet-34 + ASPP with activations qunatized to 32/4/3/2 bits. 
```
bash ./tools/dist_train.sh configs/coco_deeplabv3.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/coco_deeplabv3_4bit.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/coco_deeplabv3_3bit.py "0, 1, 2, 3" 
bash ./tools/dist_train.sh configs/coco_deeplabv3_2bit.py "0, 1, 2, 3" 
```

#### Evaluate Models

Example: evaluate VGG9 (SNN) with T = 15, 7, 3
```
bash ./tools/dist_test.sh configs/coco_deeplabv1_T15.py './workdir/coco_deeplabv1_4bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/coco_deeplabv1_T7.py './workdir/coco_deeplabv1_3bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/coco_deeplabv1_T3.py './workdir/coco_deeplabv1_2bit/best_mIoU.pth' "0, 1, 2, 3" 
```
Example: evaluate ResNet-34 + ASPP (SNN) with T = 15, 7, 3
```
bash ./tools/dist_test.sh configs/coco_deeplabv3_T15.py './workdir/coco_deeplabv3_4bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/coco_deeplabv3_T7.py './workdir/coco_deeplabv3_3bit/best_mIoU.pth' "0, 1, 2, 3" 
bash ./tools/dist_test.sh configs/coco_deeplabv3_T3.py './workdir/coco_deeplabv3_2bit/best_mIoU.pth' "0, 1, 2, 3" 
```

#### Checkpoints for Quantized Models

|       Model       |   4-bit   |   3-bit  |   2-bit  |
|     :-------:     | --------- | -------- | -------- | 
|     VGG-9         | [coco_deeplabv1_4bit](https://drive.google.com/file/d/1PsofqHqQjaFeqtyiPrIr-ETt_X-5Cako/view?usp=drive_link)  | [coco_deeplabv1_3bit](https://drive.google.com/file/d/1GDJ75JGEAx9ofnBMAcaedM7096zHX08t/view?usp=drive_link) | [coco_deeplabv1_2bit](https://drive.google.com/file/d/1IF_pAGAD1l9ViI22SovPt4aPli0L3pHx/view?usp=drive_link) | 
| ResNet-34 + ASPP  | [coco_deeplabv3_4bit](https://drive.google.com/file/d/112EEmc8QrfsVTLG_Cga3E5EDi0lch57e/view?usp=drive_link)  | [coco_deeplabv3_3bit](https://drive.google.com/file/d/144KBnPBN_GYentw9C0Xm_EC_zG6S9oXC/view?usp=drive_link) | [coco_deeplabv3_2bit](https://drive.google.com/file/d/13c4usajtQy2BapSj_MKcLwpfwciIX0kd/view?usp=drive_link) | 
