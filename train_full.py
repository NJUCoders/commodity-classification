import argparse
import os

import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau

import load_dataset
from net.load_net import load_net


def train(model_out_dir, epoch_size, batch_size, train_continue):
    net = load_net("resnet152")
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=3)
    init_epoch = 0

    if train_continue:
        checkpoint = torch.load(train_continue)
        init_epoch = checkpoint["epoch"] - 1
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("CUDA is available?", torch.cuda.is_available())

    print("Start to load train data ...", end=" ")
    train_data = load_dataset.load_from_pickle()  # 读取数据集
    # trainset, testset = torch.utils.data.random_split(train_data, [len(train_data) - int(len(train_data) / 10), int(
    #    len(train_data) / 10)])  # 将数据集随机划分为训练集与测试集，比例：[9:1]
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 读取训练集
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)  # 读取验证集
    print("Finished")

    # 开始训练
    print("Start to train ...")
    for epoch in range(init_epoch, init_epoch + epoch_size):
        running_loss = 0.0
        running_loss_ = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 训练集训练误差
            if i % 20 == 19:
                print('[%d, %5d] train loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss_ = running_loss
                running_loss = 0.0
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            # running_loss = 0.0

        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for data in testloader:
        #         images, labels = data
        #         images = images.cuda()
        #         labels = labels.cuda()
        #         outputs = net(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        # 
        # test_acc = 100 * correct / total  # 测试集泛化精度
        # scheduler.step(test_acc)
        # print('Accuracy of the network on the 10000 test images: %d %%' % test_acc)

        if not os.path.exists(model_out_dir):
            os.mkdir(model_out_dir)
        file_path = f"{model_out_dir}/model-{epoch + 1}-{running_loss_}.pth"
        state = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, file_path)
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch_size", default=100, help="set the epoch size (default: 100)")
    parser.add_argument("-b", "--batch_size", default=200, help="set the batch size (default: 200)")
    parser.add_argument("-t", "--train_continue", help="set the pretrained model path to continue train (optional)")
    parser.add_argument("-mo", "--model_out_dir", default="model",
                        help="set the model out path, (default: model)")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()
    train(model_out_dir=args.model_out_dir, epoch_size=args.epoch_size, batch_size=args.batch_size,
          train_continue=args.train_continue)
