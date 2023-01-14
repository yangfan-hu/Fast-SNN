import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from models.quant_layer import QuantReLU

import models

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet20')
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
parser.add_argument('--bit', default=2, type=int, help='the bit-width of the quantized network')

best_prec = 0
args = parser.parse_args()

def main():
    global args, best_prec
    use_gpu = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    print('=> Building model...')
    model=None
    model = models.__dict__[args.arch](bit=args.bit)
    net = models.__dict__[args.arch](bit=args.bit)
    snn = models.__dict__[args.arch](spike=True, bit=args.bit)  
    criterion = nn.CrossEntropyLoss()
    
    if use_gpu:
        model = model.cuda()
        net = net.cuda()
        snn = snn.cuda()
        # model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        
        

    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/'+str(args.arch)+'_'+str(args.bit)+'bit_ft'
    if not os.path.exists(fdir):
        os.makedirs(fdir)
        
    if not args.init:
        args.init = 'result/'+str(args.arch)+'_'+str(args.bit)+'bit/model_best.pth.tar'

    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            snn.load_state_dict(checkpoint['state_dict'])
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No pre-trained model found !')
            exit()

    print('=> loading cifar10 data...')
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
        validate(testloader, snn, criterion)
        model.show_params()
        return
    
    
    ##prepraration starts
    duration =  2**args.bit - 1    
    num_layers = 20
    num_blocks = (num_layers - 2) // 6    

    model.eval()
    snn.eval()
    
    best_acc = validate(testloader, snn, nn.CrossEntropyLoss())
    
    bypass_blocks(model, num_blocks)
    model.layer4.idem = True
    
    bypass_blocks(snn, num_blocks)
    snn.layer4.idem = True
    criterion = nn.MSELoss()
    
    ###preparation ends
    

    
    ##
    for layer_id in range(num_layers - 4):
        
        segment_id = layer_id // 2 // num_blocks + 1
        block_id = layer_id // 2% num_blocks 
        is_odd = layer_id % 2
        print('=======We are tuning Layer %d Segment %d Block %d==========' %(layer_id, segment_id, block_id))
        
        #set reference
        m = getattr(model, 'layer' + str(segment_id))
        m = getattr(m, str(block_id))
        m.idem = False
        if is_odd:
            m.inter = False
        else:
            m.inter = True
            
        #set tuner
        
        
        m = getattr(net, 'layer' + str(segment_id))
        m = getattr(m, str(block_id))
        if is_odd:
            tuner = m.part2
        else:
            tuner = m.part1
        tuner.idem = False
        tuner.relu.act_alpha.requires_grad_(False)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, tuner.parameters()),lr=1e-3, momentum=0.9, weight_decay=1e-4)


        #backup states of current block
        m = getattr(snn, 'layer' + str(segment_id))
        m = getattr(m, str(block_id))        
        record = m.state_dict()        
        for k, v in record.items():
            record[k] = v.cpu()        
        
        
        for epoch in range(1):
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
                
                m = getattr(snn, 'layer' + str(segment_id))
                m = getattr(m, str(block_id))
                m.idem = True
                
                in_maps = snn(input)
                
                if is_odd:
                    part1 = m.part1
                    part2 = m.part2
                    mid_maps = part1(in_maps)
                    out_maps = part2(mid_maps, in_maps)
                    in_maps = in_maps.sum(1).div(duration)
                    mid_maps = mid_maps.sum(1).div(duration)
                    out_maps = out_maps.sum(1).div(duration)
                    output = tuner(mid_maps, in_maps)     
                else:
                    part1 = m.part1
                    out_maps = part1(in_maps)
                    in_maps = in_maps.sum(1).div(duration)
                    out_maps = out_maps.sum(1).div(duration)
                    output = tuner(in_maps)
                output.data = out_maps.data
                
                loss = criterion(output, target_map)
                losses.update(loss.item(), input.size(0))                
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                #update weights
                if is_odd:
                    part2.load_state_dict(tuner.state_dict())
                else:
                    part1.load_state_dict(tuner.state_dict())
                    
                m = getattr(snn, 'layer' + str(segment_id))
                m = getattr(m, str(block_id))
                m.idem = False
                m.inter = False
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                           epoch, i, len(trainloader), batch_time=batch_time,
                           data_time=data_time, loss=losses))      
                # break
            
            
            for i in range(layer_id//2+1, (num_layers-4)//2):
                switch_on(snn, i, num_blocks)          
            snn.layer4.idem = False
            
            # acc = validate(testloader, snn, nn.CrossEntropyLoss())
            
            for i in range(layer_id//2+1, (num_layers-4)//2):
                switch_off(snn, i, num_blocks)              
            snn.layer4.idem = True
            
            if 1:
                print('Update...')
                # best_acc = acc
                m = getattr(snn, 'layer' + str(segment_id))
                m = getattr(m, str(block_id))        
                record = m.state_dict()        
                for k, v in record.items():
                    record[k] = v.cpu()  
                    
        #revert
        m = getattr(snn, 'layer' + str(segment_id))
        m = getattr(m, str(block_id))
        m.load_state_dict(record)
        
    #enable l4
    snn.layer4.idem = False
    
    torch.save({
        'state_dict': snn.state_dict(),
    }, os.path.join(fdir, 'model_best.pth.tar'))

    validate(testloader, snn, nn.CrossEntropyLoss())        


         
def switch_on(model, b_id, num_blocks):
    segment_id = b_id // num_blocks + 1
    block_id = b_id % num_blocks
    m = getattr(model, 'layer' + str(segment_id))
    m = getattr(m, str(block_id))
    m.idem = False
       
def switch_off(model, b_id, num_blocks):
    segment_id = b_id // num_blocks + 1
    block_id = b_id % num_blocks
    m = getattr(model, 'layer' + str(segment_id))
    m = getattr(m, str(block_id))
    m.idem = True


def bypass_blocks(model, num_blocks):
    for i in range(num_blocks):
        getattr(model.layer1, str(i)).idem = True
        getattr(model.layer2, str(i)).idem = True
    for i in range(num_blocks-1):
        getattr(model.layer3, str(i)).idem = True
        
        
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


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


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