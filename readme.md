# Vegetable birds的解题思路

- 由于数据集中图像过大，我们首先使用了python的PIL库将每张图片的大小按比例压缩到96x96的大小，对于空白处使用像素值为0填补上。
- 我们选择Pytorch框架来作为框架基础。
- 我们使用ResNet152作为CNN训练模型。
- 我们使用GTX 1080显卡，训练100个epoch，并选取其中loss最小，泛化精度最好的模型。



## 项目结构

```
.
├── cpu_predict.py # CPU预测脚本
├── easy # 下载的数据集
│   ├── data
│   ├── data.csv
│   └── test
├── gen_dataset.py # 预处理数据集脚本
├── gpu_predict.py # GPU预测脚本
├── load_dataset.py # 读取数据集脚本
├── model # 预训练好的模型
│   ├── model-87-8.477896466274615e-05.pth 
├── net # CNN网络模型
│   ├── load_net.py
│   ├── resnet152.py
│   ├── resnet.py
│   └── simplenet.py
├── requirements.txt
├── result.csv # 结果文件
├── train_full.py # 训练脚本（full）
├── train.py # 训练脚本
└── train_set.pk # 预处理数据集输出
```



## 如何运行

首先确保安装如下Python库：

```
torchvision==0.2.1
numpy==1.15.4
torch==1.0.0
Pillow==6.0.0
```

使用`python train.py`进行模型的训练。

使用`python [c|g]pu_predict.py -m {MODEL_PATH}`进行模型的预测输出。

如有问题，请加`-h`参数查看帮助。