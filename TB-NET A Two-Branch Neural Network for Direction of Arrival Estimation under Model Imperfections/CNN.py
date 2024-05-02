import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from dataset import Data_Loader_train
from dataset_verify import Data_Loader_verify
import numpy as np

# 定义参与训练的设备（cpu or gpu）
device = torch.device("cuda")

# 准备数据集
if __name__ == "__main__":
    data_dataset_train = Data_Loader_train('./TB-NET/TB-net_SNR10.mat')
    train_dataloader_train = torch.utils.data.DataLoader(dataset=data_dataset_train, shuffle=True, batch_size=32,
                                                         drop_last=True)


class conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(conv1, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2),
            nn.BatchNorm1d(num_features=out_channels, affine=False),  # 注意了，这里的批归一化层必须设置当前数据的通道数
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature(x)
        return x


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(conv, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature(x)
        return x


class C_branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_branch, self).__init__()
        self.C = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.C(x)
        return x


class R_branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(R_branch, self).__init__()
        self.R = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.R(x)
        return x


class TB(nn.Module):
    def __init__(self):
        super(TB, self).__init__()
        self.bt = nn.Sequential(
            conv1(in_channels=2, out_channels=8, kernel_size=5),
            conv1(in_channels=8, out_channels=32, kernel_size=5),
            conv1(in_channels=32, out_channels=64, kernel_size=5),
            conv1(in_channels=64, out_channels=128, kernel_size=5),
            conv1(in_channels=128, out_channels=128, kernel_size=3),
        )
        self.c = C_branch(128, 121)
        self.r = R_branch(128, 121)

    def forward(self, x):
        x1 = self.bt(x)
        x2 = self.c(x1)
        x3 = self.r(x1)
        return x2, x3


if __name__ == "__main__":
    example = TB()
    example = example.to(device)
    # 配置损失函数，C-branch用BCEloss， R-branch用MSEloss
    loss_fn_c = nn.BCELoss()
    loss_fn_c = loss_fn_c.to(device)
    loss_fn_r = nn.MSELoss()
    loss_fn_r = loss_fn_r.to(device)
    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 300
    # 添加tensorboard，用于可视化
    writer = SummaryWriter("./logs_seq")
    for i in range(epoch):
        # 配置学习速率
        learning_rate = 1e-3 * (0.9 ** (i // 30))
        optimizer = torch.optim.Adam(example.parameters(), lr=learning_rate)
        print("-----第{}轮训练开始-----".format(i + 1))
        total_train_step = 0
        # 训练步骤开始
        example.train()
        for data in train_dataloader_train:
            # 先拆分图像
            position, label_c, label_r = data
            # 将数据拷贝到device中
            position = position.to(device=device, dtype=torch.float32)
            label_c = label_c.to(device=device, dtype=torch.float32)
            label_r = label_r.to(device=device, dtype=torch.float32)
            output_c, output_r = example(position)
            output_c = output_c.to(device=device)
            output_r = output_r.to(device=device)
            # 提取最大值
            # output_c_index = torch.max(output_c)
            # output_c = torch.where(output_c == output_c_index, 1.0, 0.0)
            # # 待转换类型的PyTorch  Tensor变量带有梯度，直接将其转换为numpy数据将破坏计算图，因此numpy拒绝进行数据转换，实际上这是对开发者的一种提醒。
            # # 如果自己在转换数据时不需要保留梯度信息，可以在变量转换之前添加detach()调用。
            loss_C = loss_fn_c(output_c, label_c)
            loss_r = loss_fn_r(output_r, label_r)
            loss = 0.1 * loss_C + 2 * loss_r
            # 给梯度清零,当网络参量进行反馈时，梯度是被积累的而不是被替换掉
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
            optimizer.zero_grad()
            # 反向传播，重新计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            total_train_step = total_train_step + 1
            if total_train_step % 1000 == 0:
                print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

        if i == (epoch - 1):
            torch.save(example, '直接输出概率向量_{}.pth'.format(i))
            print("模型已保存")
        writer.close()
