from net.simplenet import SimpleNet
from net.resnet import ResNet
from net.resnet152 import ResNet152


def load_net(net_name):
    if net_name == "simplenet":
        net = SimpleNet()
    elif net_name == "resnet":
        net = ResNet()
    elif net_name == "resnet152":
        net = ResNet152()
    else:
        net = None
    return net

