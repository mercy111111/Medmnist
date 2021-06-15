import torch
import torch.nn as nn
from torch.nn import functional as F

class basic_block_resnet18_1(nn.Module):
	'''  第一类基础块  '''
	def __init__(self,in_channels):
		super(basic_block_resnet18_1,self).__init__()
		self.layer12 = nn.Sequential(
			nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
			)

	def forward(self,x):
		out = self.layer12(x)
		out = F.relu(out + x)
		return out


class basic_block_resnet18_2(nn.Module):
	'''第二类基础块'''
	def  __init__(self,in_channels,out_channels):
		super(basic_block_resnet18_2,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2)
			)
		self.layer23 = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
			)

	def forward(self,x):
		out1 = self.layer1(x)
		out2 = self.layer23(x)
		out = F.relu(out1+out2)
		return out


class ResNet_18(nn.Module):
	'''18 resnet'''
	def __init__(self, in_channels,numclass):
		super(ResNet_18,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
			)
		self.reslayer = nn.Sequential(
			basic_block_resnet18_1(64),
			basic_block_resnet18_1(64),
			basic_block_resnet18_2(64,128),
			basic_block_resnet18_1(128) ,
			basic_block_resnet18_2(128,256),
			basic_block_resnet18_1(256),
			basic_block_resnet18_2(256,512),
			basic_block_resnet18_1(512)
			)

		self.avgpool = nn.AvgPool2d(2,2,padding=1)
		self.connect = nn.Linear(512,numclass)

	def forward(self,x):
		out = self.layer1(x)
		out = self.reslayer(out)
		out = self.avgpool(F.relu(out))
		out = out.view(out.size(0),-1)
		out = self.connect(out)
		out = F.softmax(out,dim=1)
		return out



def ResNet18(in_channels, num_classes):
    return ResNet_18(in_channels,num_classes)


















class ResNet18BasicBlock_1(nn.Module):
	def __init__(self,in_channels):
		super(ResNet18BasicBlock_1,self).__init__()

		self.layer12 = nn.Sequential(
			nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
			)

	def forward(self,x):
		out = self.layer12(x)
		out = F.relu(out + x)
		return out



class ResNet18BasicBlock_2(nn.Module):
	def  __init__(self,in_channels,out_channels):
		super(ResNet18BasicBlock_2,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2)
			)
		self.layer23 = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
			)

	def forward(self,x):
		out1 = self.layer1(x)
		out2 = self.layer23(x)
		out = F.relu(out1+out2)
		return out



class Resnet_18bd(nn.Module):
	def __init__(self, numclass):
		super(Resnet_18bd,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
			)
		self.reslayer = nn.Sequential(
			ResNet18BasicBlock_1(64),
			ResNet18BasicBlock_1(64),
			ResNet18BasicBlock_2(64,128),
			ResNet18BasicBlock_1(128) ,
			ResNet18BasicBlock_2(128,256),
			ResNet18BasicBlock_1(256),
			ResNet18BasicBlock_2(256,512),
			ResNet18BasicBlock_1(512)
			)

		self.avgpool = nn.AvgPool2d(2,2,padding=1)
		self.connect = nn.Linear(512,numclass)

	def forward(self,x):
		out = self.layer1(x)
		out = self.reslayer(out)
		out = self.avgpool(F.relu(out))
		out = out.view(out.size(0),-1)
		out = self.connect(out)
		out = F.softmax(out,dim=1)
		return out



#def ResNet18(in_channels, num_classes):
    #return Resnet_18bd(num_classes)



class ResNet50DownBlock(nn.Module):
    def __init__(self, in_planes, outs, stride=1):
        super(ResNet50DownBlock, self).__init__()
        # out1, out2, out3 = outs
        # print(outs)
        self.conv1 = nn.Conv2d(in_planes, outs, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outs)
        self.conv2 = nn.Conv2d(outs, outs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outs)
        self.conv3 = nn.Conv2d(outs, 4*outs, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*outs)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != 4*outs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, 4*outs,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(4*outs)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_50bd(nn.Module):
    def __init__(self, block,in_channels=1, num_classes=2):
        super(ResNet_50bd, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._construct_layer(block, 64, 3, stride=1)
        self.layer2 = self._construct_layer(block, 128, 4, stride=2)
        self.layer3 = self._construct_layer(block, 256, 6, stride=2)
        self.layer4 = self._construct_layer(block, 512, 3, stride=2)
        self.linear = nn.Linear(512 * 4, num_classes)

    def _construct_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * 4
        return nn.Sequential(*layers)



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet50(in_channels, num_classes):
    return ResNet_50bd(ResNet50DownBlock,in_channels, num_classes)


