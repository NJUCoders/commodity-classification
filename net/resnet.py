# 读取pytorch自带的resnet-101模型,因为使用了预训练模型，所以会自动下载模型参数
# model=models.resnet101(pretrained=True)
#
# # 对于模型的每个权重，使其不进行反向传播，即固定参数
# for param in model.parameters():
#     param.requires_grad = False
# # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层fc
# for param in model.fc.parameters():
#     param.requires_grad = True
#
# # 假设要分类数目是200
# class_num = 200
#
# # 获取fc层的输入通道数
# channel_in = model.fc.in_features
#
# # 然后把resnet-101的fc层替换成200类别的fc层
# model.fc = nn.Linear(channel_in, class_num)
#
# model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    '''
    实现子Module:ResidualBlock
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    '''
    实现主Module:ResNet34
    ResNet34包含多个layer，每个layer又包含多个reesidal block
    用子module实现residual block，用_make_layer函数实现layer
    '''

    def __init__(self, num_classes=200):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 重复的layer，分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer，包含多个residual
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
