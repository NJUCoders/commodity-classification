from net.simplenet import SimpleNet
from net.resnet import ResNet


def load_net(net_name):
    if net_name == "simplenet":
        net = SimpleNet()
    elif net_name == "resnet":
        net = ResNet()
    else:
        net = None
    return net
