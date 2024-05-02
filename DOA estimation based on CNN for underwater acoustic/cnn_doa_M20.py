import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from dataset import Data_Loader

# 定义参与训练的设备（cpu or gpu）
if __name__ == "__main__":
    device = torch.device("cuda")

# 准备数据集
if __name__ == "__main__":
    data_dataset = Data_Loader('./CNN_DOA/CNN_SNR20.mat')
    train_dataloader = torch.utils.data.DataLoader(dataset=data_dataset, shuffle=True, batch_size=32, drop_last=True)

# 分割数据
if __name__ == "__main__":
    reg1 = []
    for data in train_dataloader:
        reg1.append(data)
    x_train, x_test = train_test_split(reg1, train_size=0.8)


class Conv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2, stride=1, padding=1, dilation=2),
            # nn.BatchNorm2d(32),  # 进行数据的归一化处理,在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            # nn.ReLU(),  # 如果没有非线性激活每一层的输出都是上层输入的线性函数,通过非线性激活后输出不再是输入的线性组合，可以逼近任意函数
            nn.MaxPool2d(kernel_size=2, stride=1, dilation=1),
        )

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class Conv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=2),
            # nn.BatchNorm2d(64),  # 进行数据的归一化处理,在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            # nn.ReLU(),  # 如果没有非线性激活每一层的输出都是上层输入的线性函数,通过非线性激活后输出不再是输入的线性组合，可以逼近任意函数
            nn.MaxPool2d(kernel_size=2, stride=1, dilation=1),
        )

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class Conv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=2),
            # nn.BatchNorm2d(128),  # 进行数据的归一化处理,在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            # nn.ReLU(),  # 如果没有非线性激活每一层的输出都是上层输入的线性函数,通过非线性激活后输出不再是输入的线性组合，可以逼近任意函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=2),
            # nn.BatchNorm2d(128),  # 进行数据的归一化处理,在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            # nn.ReLU(),  # 如果没有非线性激活每一层的输出都是上层输入的线性函数,通过非线性激活后输出不再是输入的线性组合，可以逼近任意函数
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
        )

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class CNN_doa(nn.Module):
    def __init__(self):
        super().__init__()
        self.M20 = nn.Sequential(
            Conv1(),
            Conv2(),
            Conv3(),
            nn.Flatten(),
            nn.Linear(in_features=10368, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=181),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.M20(x)
        return x1


if __name__ == "__main__":
    example = CNN_doa()
    example = example.to(device)
    loss_fn = nn.BCELoss()

    # loss_fn = tensorflow.keras.losses.SparseCategoricalCrossentropy()
    loss_fn = loss_fn.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(example.parameters(), lr=learning_rate)

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 20
    # 添加tensorboard，用于可视化
    writer = SummaryWriter("./logs_seq")

    for i in range(epoch):
        print("-----第{}轮训练开始-----".format(i + 1))
        total_train_step = 0
        # 训练步骤开始
        example.train()
        for data in train_dataloader:
            # 先拆分图像
            position, targets = data
            # 将数据拷贝到device中
            position = position.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            outputs = example(position)
            loss = loss_fn(outputs, targets)  # 误差的平方和再开方除2，MSEloss计算出来的结果是除过N=540的
            # 给梯度清零
            # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 重新计算梯度，更新参数
            optimizer.step()
            total_train_step = total_train_step + 1
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss5", loss.item(), total_train_step)

        if i == (epoch - 1):
            torch.save(example, 'example_SNR20{}.pth'.format(i))
            print("模型已保存")
        writer.close()
