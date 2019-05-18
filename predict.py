import csv
import os

import numpy
import torch
import torch.cuda
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from train import Net

net = Net()
model_dir = "model"
checkpoint = torch.load(f"{model_dir}/model-74-91.01.pth")
net.load_state_dict(checkpoint['state_dict'])


def predict_image(image_path):
    print("Prediction in progress")
    image = numpy.asarray(Image.open(image_path).resize((32, 32)))

    # Define transformations for the image
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

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
    return index


def predict():
    # 给出要预测图片文件位置，返回预测CategoryId（index）
    index = predict_image("easy/data/1ce6baf8-25c1-4328-a268-1129b98c600c.jpg")
    print(index)


if __name__ == '__main__':
    file_test = open('result.csv', 'w', newline='')
    csv_writer = csv.writer(file_test)
    csv_writer.writerow(["ImageName", "CategoryId"])
    for filename in os.listdir("easy/test"):
        csv_writer.writerow((filename, predict_image(f"easy/test/{filename}")))
