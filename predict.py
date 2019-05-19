import csv
import argparse
import os
import sys

import numpy
import torch
import torch.cuda
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from net.load_net import load_net


test_transformations = transforms.Compose([
    transforms.ToTensor()
])


def load_trained_net(model_path):
    print("Begin to load pre-trained net ... ", end="")
    net = load_net("simplenet")
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    print("Finished.")
    return net


def predict_image(net, image_path: str):
    image = numpy.asarray(Image.open(image_path).resize((32, 32)))

    # Define transformations for the image
    transformation = test_transformations

    # 预处理图像
    image_tensor = transformation(image)

    # 额外添加一个批次维度，因为PyTorch将所有的图像当做批次
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # 将输入变为变量
    input = Variable(image_tensor)

    # 预测图像的类
    output = net(input)
    index = output.data.numpy().argmax()
    return index + 1


def predict(net, outfile_path: str):
    file_test = open(outfile_path, 'w', newline='')
    csv_writer = csv.writer(file_test)
    csv_writer.writerow(["ImageName", "CategoryId"])
    for i, filename in enumerate(os.listdir("easy/test")):
        print("Prediction in progress:", i)
        csv_writer.writerow((filename, predict_image(net=net, image_path=f"easy/test/{filename}")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="set the pretrained model path")
    parser.add_argument("-o", "--outfile_path", default="result.csv", help="set the output file path, default to result.csv")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()
    predict(net=load_trained_net(args.model_path), outfile_path=args.outfile_path)



