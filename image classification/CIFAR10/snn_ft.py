import argparse
import os
import time
import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import copy

from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res20')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('-n', '--num_epochs', default=1, type=int, help='number of epochs for fine-tuning')
parser.add_argument('--force', action='store_true', help='Force tuner to always update weights')
best_prec = 0
args = parser.parse_args()

def main():

    global args, best_prec
    use_gpu = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    print('=> Building model...')
    model=None
    
    float = True if args.bit == 32 else False
    if args.arch == 'alex':
        model = AlexNet(float=float)
        snn = S_AlexNet(T = 2**args.bit - 1)
    elif args.arch == 'vgg11':
        model = VGG11(float=float)
        snn = S_VGG11(T = 2**args.bit - 1)
    else:
        print('Architecture not support!')
        return
    if not float:
        for m in model.modules():
            #Ouroboros-------determine quantization
            #APoT quantization for weights, uniform quantization for activations
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                #weight quantization, use APoT
                m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
            if isinstance(m, QuantReLU):
                #activation quantization, use uniform
                m.act_grid = build_power_value(args.bit)
                m.act_alq = act_quantization(b=args.bit, grid=m.act_grid, power=False)          
        for m in snn.modules():
            #Ouroboros-------determine quantization
            #APoT quantization for weights, uniform quantization for activations
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                #weight quantization, use APoT
                m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
            if isinstance(m, QuantReLU):
                #activation quantization, use uniform
                m.act_grid = build_power_value(args.bit)
                m.act_alq = act_quantization(b=args.bit, grid=m.act_grid, power=False)                    


    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/'+str(args.arch)+'_'+str(args.bit)+'bit' + '_ft'
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    
    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            if use_gpu:
                print('use gpu')
                checkpoint = torch.load(args.init, map_location='cpu')
                model = model.cuda()
                snn = snn.cuda()
                cudnn.benchmark = True                
            else:
                print('use cpu')
                checkpoint = torch.load(args.init, map_location='cpu')
                criterion = nn.CrossEntropyLoss()
                
            #Remove DataParallel wrapper 'module' 
            for name in list(checkpoint['state_dict'].keys()):
                checkpoint['state_dict'][name[7:]] = checkpoint['state_dict'].pop(name)
                                
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            snn.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            print('No pre-trained model found !')
            exit()

    print('=> loading cifar10 test data...')
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(trainloader, model, nn.CrossEntropyLoss())
        model.show_params()
        return
    
    if args.arch == 'alex':
        num_layers = 7
        flatten_id = 6
    elif args.arch == 'vgg11':
        num_layers = 11
        flatten_id = 9
    

    model.eval()
    snn.eval()
    criterion = nn.MSELoss()
    duration =  2**args.bit - 1   
    
    best_acc = 0
    acc = 0
    
    if not args.force:
        best_acc = validate(trainloader, snn, nn.CrossEntropyLoss())
    
    validate(testloader, snn, nn.CrossEntropyLoss())
    # start_time = time.time()
    
    #bypass layer 2 to end
    for i in range(2, num_layers + 1):
        getattr(model, 'layer' + str(i)).idem = True
        getattr(snn, 'layer' + str(i)).idem = True
    #flat
    getattr(model, 'flat').idem = True
    getattr(snn, 'flat').idem = True   
    
    #####fine tuning##### 
    #layer 2 to L - 1
    for layer_id in range(2,num_layers):
        print('=======We are tuning Layer %d ==========' %layer_id)

        tuner = copy.deepcopy(getattr(model, 'layer' + str(layer_id)))
        tuner.idem = False
        # tuner.train()
        
        tuner.block[2].act_alpha.requires_grad_(False) #do not tune scalar factor
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, tuner.parameters()),lr=1e-3, momentum=0.9, weight_decay=5e-4)
        
        #best_loss = 65535
        getattr(model, 'layer' + str(layer_id)).idem = False
        if layer_id == flatten_id:
            getattr(model, 'flat').idem = False
            getattr(snn, 'flat').idem = False
            
            
        record = getattr(snn, 'layer' + str(layer_id)).state_dict()
        for k, v in record.items():
            record[k] = v.cpu()
            
        
        for epoch in range (args.num_epochs):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()    
            end = time.time()
            
            for i, (input, target) in enumerate(trainloader):
                data_time.update(time.time() - end)
                
                input = input.to(args.device)
                target = target.to(args.device)
                with torch.no_grad():
                    target_map = model(input)
                
                getattr(snn, 'layer' + str(layer_id)).idem = True
                in_maps = snn(input) 
                getattr(snn, 'layer' + str(layer_id)).idem = False
                out_maps = getattr(snn, 'layer' + str(layer_id))(in_maps)
                #eval
                in_maps = in_maps.sum(1).div(duration)
                out_maps = out_maps.sum(1).div(duration)
                
                # in_maps.requires_grad_()
                output = tuner(in_maps)
                output.data = out_maps.data
                
                
                loss = criterion(output, target_map)
                losses.update(loss.item(), input.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()         
                
                #update weights
                getattr(snn, 'layer' + str(layer_id)).load_state_dict(tuner.state_dict(), strict=False)
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                           epoch, i, len(trainloader), batch_time=batch_time,
                           data_time=data_time, loss=losses))  

            ###val###
            if not args.force:
                for i in range(layer_id + 1, num_layers + 1):
                    getattr(snn, 'layer' + str(i)).idem = False
                if layer_id < flatten_id:
                    getattr(snn, 'flat').idem = False
                
                acc = validate(trainloader, snn, nn.CrossEntropyLoss()) 
                
                for i in range(layer_id + 1, num_layers + 1):
                    getattr(snn, 'layer' + str(i)).idem = True
                if layer_id < flatten_id:
                    getattr(snn, 'flat').idem = True                    

            if acc > best_acc or args.force:
                record = getattr(snn, 'layer' + str(layer_id)).state_dict()
                for k, v in record.items():
                    record[k] = v.cpu()      
                best_acc = acc
                print('Updated')
            else:
                #restore previous weights
                print('Do Nothing')
        #revert to best
        getattr(snn, 'layer' + str(layer_id)).load_state_dict(record) 
                  
    getattr(snn, 'layer' + str(num_layers)).idem = False    
    # torch.save({
    #     'state_dict': snn.state_dict(),
    # }, os.path.join(fdir, 'checkpoint.pth'))

    
    validate(testloader, snn, nn.CrossEntropyLoss())
    # print(time.time()-start_time)
    
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


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(args.device), target.to(args.device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()