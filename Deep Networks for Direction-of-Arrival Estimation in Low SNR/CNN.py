import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, Linear, Dropout, Sigmoid, Flatten
from torch.utils.tensorboard import SummaryWriter
from dataset import Data_Loader

# 定义参与训练的设备（cpu or gpu）
device = torch.device("cuda")

# 准备数据集
if __name__ == "__main__":
    # data_dataset = Data_Loader('./train/matlab.mat')
    # train_dataloader = torch.utils.data.DataLoader(dataset=data_dataset, batch_size=32, drop_last=True)
    data_dataset = Data_Loader('./DNN/DNN_SNR20.mat')
    train_dataloader = torch.utils.data.DataLoader(dataset=data_dataset, shuffle=True, batch_size=64, drop_last=True)  #

# 分割数据
if __name__ == "__main__":
    reg1 = []
    for i, data in enumerate(train_dataloader):
        reg1.append(1)
        reg1[i] = data#
        # print(reg[i])
    x_train, x_test = train_test_split(reg1, train_size=0.9)

    # length数据的长度
    train_data_size = len(train_dataloader)
    # test_data_size = len(test_dataset)
    # 输出数据集长度，检验数据集是否正确导入
    print("训练数据集的长度为：{}".format(train_data_size))


class Conv(nn.Module):
    def __init__(self, num_input_channels, output_channels, kernel_size, stride, padding):
        super().__init__()
        self.module1 = Sequential(
            # 由于输入的是16*16的图像，导入输出图像大小的公式得到（16-3）/2=6.5，认为是走了6步，0.5步舍去，所以就造成了信息的丢失
            # padding =1 就是在原图上下左右都增加一层0，原图尺寸变为（18-3）/2=7.5，走7步，把原图的信息全部保留下来
            Conv2d(num_input_channels, output_channels, kernel_size, stride, padding=padding, bias=False),
            # 3×3卷积核，进行卷积，得到通道数为16
            nn.BatchNorm2d(output_channels),  # 进行数据的归一化处理,在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            ReLU(),  # 如果没有非线性激活每一层的输出都是上层输入的线性函数,通过非线性激活后输出不再是输入的线性组合，可以逼近任意函数
        )

    def forward(self, x):
        x1 = self.module1(x)
        return x1


# 收缩路径重复部分 denseblock+MaxPool2d
class FC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module1 = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Dropout(0.3),
        )

    def forward(self, x):
        x1 = self.module1(x)
        return x1


# 扩张路径重复部分 denseblock+MaxPool2d
class last_layer(nn.Module):
    def __init__(self, in_channels, G):
        super().__init__()
        self.module1 = Sequential(
            Linear(in_channels, 2 * G + 1),
            Sigmoid(),
        )

    def forward(self, x):
        x1 = self.module1(x)
        return x1


# -------------搭建DenseNet网络-----------------------------
class CNN(nn.Module):
    def __init__(self,
                 conv_in_channels: int = 3,
                 conv_out_channels: int = 256,
                 FC_out_channels: int = 4096,
                 G: int = 90):
        super().__init__()
        self.module = Sequential(
            Conv(conv_in_channels, conv_out_channels, 3, 2, 0),
            Conv(conv_out_channels, conv_out_channels, 2, 1, 0),
            Conv(conv_out_channels, conv_out_channels, 2, 1, 0),
            Conv(conv_out_channels, conv_out_channels, 2, 1, 0),
            Flatten(),
            FC(256 * 4 * 4, FC_out_channels),  # 输入的是flatten之后的，需要根据阵元数随时调整，比如10阵元就得是256*1*1
            FC(FC_out_channels, int(FC_out_channels / 2)),
            FC(int(FC_out_channels / 2), int(FC_out_channels / 4)),
            last_layer(int(FC_out_channels / 4), G),
        )

    def forward(self, x):
        x1 = self.module(x)
        return x1


if __name__ == "__main__":
    #
    # photo = torchvision.models.CNN()
    # tw.draw_model(photo, [1, 3, 224, 224])

    # 创建模型
    example = CNN()
    example = example.to(device)

    # 损失函数,二进制交叉熵函数
    loss_fn = nn.BCELoss()
    loss_fn = loss_fn.to(device)
    # 损失函数,均方误差
    loss_fn1 = nn.MSELoss()
    loss_fn1 = loss_fn.to(device)

    # 优化器,利用优化器可以从反向传播后的梯度入手，对数据进行优化
    # 举例： learning_rate = 0.01 = 1e-2
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(example.parameters(), lr=learning_rate)

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 100
    # 添加tensorboard，用于可视化
    writer = SummaryWriter("./logs_seq")
    # 计数变量
    n = 0

    for i in range(epoch):
        print("-----第{}轮训练开始-----".format(i + 1))
        total_train_step = 0
        # 训练步骤开始
        example.train()
        for reg in train_dataloader:
            # 先拆分图像
            data, labels = reg
            # 将数据拷贝到device中
            data = data.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            # targets = targets.to(device)
            outputs = example(data)
            loss = loss_fn(outputs, labels)
            # 给梯度清零
            # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 重新计算梯度，更新参数
            optimizer.step()
            total_train_step = total_train_step + 1
            # if total_train_step % 500 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss1", loss.item(), total_train_step)

        # 测试开始
        # example.eval()
        # with torch.no_grad():
        #     for data in x_test:
        #         n += 1
        #         reg, label = data
        #         reg = reg.to(device=device, dtype=torch.float32)
        #         label = label.to(device=device, dtype=torch.float32)
        #         outputs = example(reg)
        #         loss = loss_fn1(outputs, label)
        #         total_test_step = total_test_step + 1
        #         print('模型在测试集上的loss：{}'.format(loss))
        #         writer.add_scalar("test_loss", loss.item(), total_test_step)

        total_test_step = total_test_step + 1
        # 保存每一次训练之后的模型和数据
        if i==epoch-1:
            torch.save(example, 'example_SNR20{}.pth'.format(i))  # 保存模型
        print("模型已保存")
        writer.close()
