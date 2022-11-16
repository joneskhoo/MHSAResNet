import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.metrics
import ignite.contrib.handlers

import random
import numpy as np

DATA_DIR='./data'

NUM_CLASSES = 10
NUM_WORKERS = 2
BATCH_SIZE = 32
EPOCHS = 100

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", DEVICE)

import os.path
def check_mnist_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'mnist/train_data.pt')
        torch.save(train_label,path_data + 'mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'mnist/test_data.pt')
        torch.save(test_label,path_data + 'mnist/test_label.pt')
    return path_data#, trainset, testset

data_path=check_mnist_dataset_exists()

train_data=torch.load(data_path+'mnist/train_data.pt')
train_label=torch.load(data_path+'mnist/train_label.pt')
test_data=torch.load(data_path+'mnist/test_data.pt')
test_label=torch.load(data_path+'mnist/test_label.pt')

print(train_data.data.size())
print(test_data.data.size())

resize = transforms.Resize((32, 32))
print(train_data.shape)
train_loader = DataLoader(list(zip(resize(train_data.unsqueeze(1)),train_label)), batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(list(zip(resize(test_data.unsqueeze(1)),test_label)), batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True)

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = 8#heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.gating_param = nn.Parameter(torch.ones(self.heads))
        self.width = width


    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out



# defining resnet models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,size=32):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        #if size>0:
        #    self.botnet = BottleneckAttention(dim=planes, fmap_size=(size,size))
        #self.size=size

    def forward(self, x):
       
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #print(out.shape)
        #if self.size>0:
        #    out = self.botnet(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Botneck2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,size=32):
        super(Botneck2, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                       stride=1, padding=1, bias=False)
        heads=9
        q_channels = planes // heads
        #self.conv2=SelfAttention2d(planes, planes, q_channels, q_channels, heads, RelativePosEnc, 0.0)
        self.size=4
        if planes == 256:
            self.size=8
        self.conv2=MHSA(planes, width=self.size, height=self.size, heads=8)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
       
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x)+self.gamma * out
        out = F.relu(out)
       
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # This is the "stem"
        # For CIFAR (32x32 images), it does not perform downsampling
        # It should downsample for ImageNet
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # four stages with three downsampling
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        
       
        self.layer4 = self._make_layer(Botneck2, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.max1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max2=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max3=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def _make_layer(self, block, planes, num_blocks, stride,size=32):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        #out=self.max1(out)
        out = self.layer2(out)
        #out=self.max2(out)
        
        #jones
        out = self.layer3(out)
        #out=  self.max3(out)
        out = self.layer4(out)

        #out = self.layer5(out)
       
        #print("out shape",out.shape)
        #out = F.avg_pool2d(out, 2)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out


def ResNet18():
    #return ResNet(Bottleneck, [2, 2, 2, 2])
    return ResNet(BasicBlock, [2, 2, 2, 2])



def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test_resnet18():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

import os
def load_checkpoint(model, optimizer, lr_scheduler, filename='convit.pt'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if True:#os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        #model=torch.load(filename)
        #model.to(DEVICE)
        checkpoint = torch.load(filename+'_a')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer=checkpoint['optimizer']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #lr_scheduler=checkpoint['lr_scheduler']


        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, lr_scheduler


torch.no_grad()
def init_linear(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)


class History:
    def __init__(self):
        self.values = defaultdict(list)

    def append(self, key, value):
        self.values[key].append(value)

    def reset(self):
        for k in self.values.keys():
            self.values[k] = []

    def _begin_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def _end_plot(self, ylabel):
        self.ax.set_xlabel('epoch')
        self.ax.set_ylabel(ylabel)
        plt.show()

    def _plot(self, key, line_type='-', label=None):
        if label is None: label=key
        xs = np.arange(1, len(self.values[key])+1)
        self.ax.plot(xs, self.values[key], line_type, label)

    def plot(self, key):
        self._begin_plot()
        self._plot(key, '-')
        self._end_plot(key)

    def plot_train_val(self, key):
        self._begin_plot()
        self._plot('train ' + key, '.-', 'train')
        self._plot('val ' + key, '.-', 'val')
        self.ax.legend()
        self._end_plot(key)

def separate_parameters(model):
    # biases, and batchnorm weights will not be decayed for regularization
    #parameters_decay = set()
    parameters_decay = list()
    #parameters_no_decay = set()
    parameters_no_decay = list()
    modules_weight_decay = (nn.Linear, nn.Conv2d)
    modules_no_weight_decay = (nn.BatchNorm2d,)

    for m_name, m in model.named_modules():
        for param_name, param in m.named_parameters():
            full_param_name = f"{m_name}.{param_name}" if m_name else param_name

            if isinstance(m, modules_no_weight_decay):
                parameters_no_decay.append(full_param_name)
            elif param_name.endswith("bias"):
                parameters_no_decay.append(full_param_name)
            elif param_name.endswith("gamma"):
                parameters_no_decay.append(full_param_name)
            elif isinstance(m, modules_weight_decay):
                parameters_decay.append(full_param_name)

    return parameters_decay, parameters_no_decay

from collections import OrderedDict

def get_optimizer(model, learning_rate, weight_decay):
    #param_dict = {pn: p for pn, p in model.named_parameters()}
    list_of_tuples = [(pn, p) for pn, p in model.named_parameters()]
    param_dict  = OrderedDict(list_of_tuples)
    parameters_decay, parameters_no_decay = separate_parameters(model)
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in parameters_decay], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in parameters_no_decay], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=learning_rate)
    #optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.9)

    return optimizer

model=ResNet18()
model.apply(init_linear);
model.to(DEVICE);
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
print(model)
optimizer =get_optimizer(model, learning_rate=1e-6, weight_decay=1e-2)
lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                             steps_per_epoch=len(train_loader), epochs=100)
start_epoch=1

print("Number of parameters:", sum(p.numel() for p in model.parameters()))
print("Start Epoch",start_epoch)

loss = nn.CrossEntropyLoss()

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

def evaluate():
    model.eval()
    with torch.no_grad():

        correct, total = 0, 0
        test_loss = 0.0
        nbatch=0
        for batch in test_loader:
            nbatch+=1
            x, y = batch     
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            y_hat = model(x)
            #loss_v = loss(y_hat, y) / len(x)
            loss_v = loss(y_hat, y)

            test_loss += loss_v.detach().cpu().item()

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        f = open("result.txt", "a")
        test_loss=test_loss/nbatch
        print(f"\nTest loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.6f}%\n")
        f.write(f"Test loss: {test_loss:.2f}\n")
        f.write(f"Test accuracy: {correct / total * 100:.6f}%\n")
        val_acc = correct / total*100
        f.close()
        val_loss_list.append(test_loss)
        val_acc_list.append(val_acc)
       
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

import copy as cp

for epoch in range(EPOCHS):
    model.train() 

    correct, total = 0, 0
    train_loss = 0.0
    
    nbatch=0
    for batch in train_loader:
        nbatch+=1
        x, y = batch
                
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        
        y_hat = model(x)
        
        #loss_v = loss(y_hat, y) / len(x)
        loss_v = loss(y_hat, y)


        train_loss += loss_v.detach().cpu().item()
        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total += len(x)

        optimizer.zero_grad()
        loss_v.backward()
        optimizer.step()
        lr_scheduler.step()

     
    f = open("result.txt", "a")
    lr=get_lr(optimizer)
    train_loss=train_loss/nbatch
    print(f"Epoch {epoch + 1}/{EPOCHS} loss: {train_loss} lr:{lr}")
    f.write(f"Epoch {epoch + 1}/{EPOCHS} loss: {train_loss} lr:{lr}\n")   
    train_acc = correct / total*100
    print(f"Train accuracy: {correct / total * 100:.6f}%")
    f.write(f"Train accuracy: {correct / total * 100:.6f}%\n")
    f.close()
    train_loss_list.append(train_loss)
    
    train_acc_list.append(train_acc)
    evaluate()
    
    if epoch % 10 ==0:
        torch.save(model,'convit.pt')
        state = {'epoch': epoch + 1, 'state_dict':cp.deepcopy(model.state_dict()),
             'optimizer': cp.deepcopy(optimizer.state_dict()),  'lr_scheduler':cp.deepcopy(lr_scheduler.state_dict())}
        torch.save(state, 'convit.pt_a')


print(train_loss_list)
print(train_acc_list)
print(val_loss_list)
print(val_acc_list)


