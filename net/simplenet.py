import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(192, 200, 1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(6, stride=1, padding=0)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.drop3 = nn.Dropout2d(0.5)
        self.fc = nn.Linear(200, 200)

    def forward(self, x):
        data = self.drop1(x)
        conv1 = F.relu(self.conv1(data))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = self.drop2(F.relu(self.conv3(conv2)))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))
        conv6 = self.drop3(F.relu(self.conv6(conv5)))
        conv7 = F.relu(self.conv7(conv6))
        conv8 = F.relu(self.conv8(conv7))
        conv9 = F.relu(self.conv9(conv8))
        pool = self.pool(conv9)
        x = pool.view(-1, 200)
        return self.fc(x)
