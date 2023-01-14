import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
import models
# import numpy as np

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)

    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    parser.add_argument('--bit', default=5, type=int, help='the bit-width of the quantized network')
    parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
    parser.add_argument('-n', '--num_epochs', default=1, type=int, help='number of epochs for fine-tuning')
    parser.add_argument('--force', action='store_true', help='Force tuner to always update weights')
    args = parser.parse_args()
    return args

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()

    # test mode, use default args for sanity test
    if args.test:
        args.opt_level = None
        args.epochs = 1
        args.start_epoch = 0
        args.arch = 'resnet50'
        args.batch_size = 64
        args.data = []
        args.sync_bn = False
        args.data.append('/data/imagenet/train/')
        args.data.append('/data/imagenet/val/')
        print("Test mode - no DDP, no apex, RN50, 10 iterations")

    if not len(args.data):
        raise Exception("error: No data set provided")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # make apex optional
    if args.distributed:
        try:
            global optimizers
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))


    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        # device = torch.device('cuda', args.gpu)
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        print('Local Rank: %d' %args.gpu)

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](spike=False, bit=args.bit)
    tuner = models.__dict__[args.arch](spike=False, bit=args.bit)
    snn = models.__dict__[args.arch](spike=True, bit=args.bit)



    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
        snn = snn.cuda().to(memory_format=memory_format)
        tuner = tuner.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()
        snn = snn.cuda()
        tuner = tuner.cuda()
        print('Job is done')

    # Scale learning rate based on global batch size
    # args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    
    ###get total layers L
    num_layers = model.num_layers
    flatten_id = model.flatten_id
    duration = 2**args.bit - 1
    #freeze clipping threshold
    for i in range(1, num_layers):
        getattr(tuner, 'layer' + str(i)).block[2].act_alpha.requires_grad_(False)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, tuner.parameters()),lr=1e-3, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    
    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        print('Starting DDP')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
        snn = torch.nn.parallel.DistributedDataParallel(snn, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
        tuner = torch.nn.parallel.DistributedDataParallel(tuner, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)


    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/'+str(args.arch)+'_'+str(args.bit)+'bit_ft'
    if not os.path.exists(fdir):
        os.makedirs(fdir)


    if args.init:
        if os.path.isfile(args.init):
            print("=> loading initialization model from {}".format(args.init))
            #load init to CPU to conserve GPU memory-> by lucian
            checkpoint = torch.load(args.init, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            model.module.show_params()
            snn.load_state_dict(checkpoint['state_dict'])
            tuner.load_state_dict(checkpoint['state_dict'])
        else:
            print('no initialization model found')
            exit()        
    else:
        print('initialization model required for tuning')
        exit()                


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_ft = nn.MSELoss()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Data loading code
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    #preparation 
    model.eval()
    snn.eval()
    
    prec1 = 0
    best_prec1 = 0
    
    if not args.force:
        [prec1, prec5] = validate(train_loader, snn, criterion)
        train_loader.reset()
        
    best_prec1 = prec1

    #tuner bypass layer 1 to end (deactivate all layers)
    for i in range(1, num_layers + 1):
        getattr(tuner.module, 'layer' + str(i)).idem = True
    getattr(tuner.module, 'flat').idem = True 
    
    #bypass layer 2 to end
    for i in range(2, num_layers + 1):
        getattr(model.module, 'layer' + str(i)).idem = True
        getattr(snn.module, 'layer' + str(i)).idem = True
        
    #flat
    getattr(model.module, 'flat').idem = True
    getattr(snn.module, 'flat').idem = True       
    
    #####fine tuning##### 
    #layer 2 to L - 1
    for layer_id in range(2,num_layers):
        if args.local_rank == 0:
            print('=======We are tuning Layer %d ==========' %layer_id)
        #enable tuner at layer l
        getattr(tuner.module, 'layer' + str(layer_id)).idem = False
        
        getattr(model.module, 'layer' + str(layer_id)).idem = False
        if layer_id == flatten_id:
            getattr(model.module, 'flat').idem = False
            getattr(snn.module, 'flat').idem = False    
        
        record = getattr(snn.module, 'layer' + str(layer_id)).state_dict()
        for k, v in record.items():
            record[k] = v.cpu()
        
        
        for epoch in range (args.num_epochs):
            batch_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            
            for i, data in enumerate(train_loader):
                input = data[0]["data"]
                # target = data[0]["label"].squeeze(-1).long()
                train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
                
                with torch.no_grad():
                    target_map = model(input)
                
                getattr(snn.module, 'layer' + str(layer_id)).idem = True
                in_maps = snn(input) 
                getattr(snn.module, 'layer' + str(layer_id)).idem = False
                out_maps = getattr(snn.module, 'layer' + str(layer_id))(in_maps)
                #eval
                in_maps = in_maps.sum(1).div(duration)
                out_maps = out_maps.sum(1).div(duration)
                
                # in_maps.requires_grad_()
                output = tuner(in_maps)
                output.data = out_maps.data
                
                
                loss = criterion_ft(output, target_map)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()         
                 
                #update weights
                getattr(snn.module, 'layer' + str(layer_id)).load_state_dict(getattr(tuner.module, 'layer' + str(layer_id)).state_dict())
                
                if i%args.print_freq == 0:
                    # Every print_freq iterations, check the loss, accuracy, and speed.
                    # For best performance, it doesn't make sense to print these metrics every
                    # iteration, since they incur an allreduce and some host<->device syncs.
        
                    # Average loss and accuracy across processes for logging
                    if args.distributed:
                        reduced_loss = reduce_tensor(loss.data)
                    else:
                        reduced_loss = loss.data
        
                    # to_python_float incurs a host<->device sync
                    losses.update(to_python_float(reduced_loss), input.size(0))
        
                    torch.cuda.synchronize()
                    batch_time.update((time.time() - end)/args.print_freq)
                    end = time.time()
        
                    if args.local_rank == 0:
                        print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Speed {3:.3f} ({4:.3f})\t'
                              'Loss {loss.val:.10f} ({loss.avg:.4f})\t'.format(
                               epoch, i, train_loader_len,
                               args.world_size*args.batch_size/batch_time.val,
                               args.world_size*args.batch_size/batch_time.avg,
                               batch_time=batch_time,
                               loss=losses))
                           
            ###val###
            #remove identity
            if not args.force:
                for i in range(layer_id + 1, num_layers + 1):
                    getattr(snn.module, 'layer' + str(i)).idem = False
                if layer_id < flatten_id:
                    getattr(snn.module, 'flat').idem = False
                
                train_loader.reset()
                [prec1, prec5] = validate(val_loader, snn, criterion)  
                
                #set identity
                for i in range(layer_id + 1, num_layers + 1):
                    getattr(snn.module, 'layer' + str(i)).idem = True
                if layer_id < flatten_id:
                    getattr(snn.module, 'flat').idem = True                    
                
            #update
            if prec1 > best_prec1 or args.force:
                record = getattr(snn.module, 'layer' + str(layer_id)).state_dict()
                for k, v in record.items():
                    record[k] = v.cpu()                
                    
                best_prec1 = prec1
                if args.local_rank == 0:
                    print('Updated')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': snn.state_dict(),
                        'best_prec1': best_prec1,
                        'best_prec5':prec5,
                        'optimizer' : optimizer.state_dict(),
                    }, True, fdir)         
            else:
                #restore previous weights
                if args.local_rank == 0:
                    print('Do Nothing')
            #reset dali iterators
            train_loader.reset()
            # val_loader.reset()
        #revert to best
        if args.local_rank == 0:
            print('Revert to best......')
        
        getattr(snn.module, 'layer' + str(layer_id)).load_state_dict(record) 
        
        #disable tuner at layer l
        getattr(tuner.module, 'layer' + str(layer_id)).idem = True
             
    getattr(snn.module, 'layer' + str(num_layers)).idem = False
    [prec1, prec5] = validate(val_loader, snn, criterion)
    
    return
    #ends here


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        if args.test:
            if i > 10:
                break

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))


    return batch_time.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        val_loader_len = int(val_loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#reducing epochs
def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 20

    if epoch >= 50:
        factor = factor + 1

    lr = args.lr*(0.1**factor)


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()