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
    data_dataset = Data_Loader("./data/matlab_M10_SNR-10.mat")
    train_dataloader = torch.utils.data.DataLoader(dataset=data_dataset, shuffle=False, batch_size=32, drop_last=False)

# 分割数据
if __name__ == "__main__":
    reg1 = []
    for data in train_dataloader:
        reg1.append(data)
    x_train, x_test = train_test_split(reg1, train_size=0.8)


class Conv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, dilation=2),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class Conv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

    def forward(self, x):
        x1 = self.conv(x)
        return x1


class CNN_doa(nn.Module):
    def __init__(self):
        super().__init__()
        self.M10 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2, stride=1, padding=1, dilation=2),
            Conv2(),
            Conv3(),
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=181)
        )

    def forward(self, x):
        x1 = self.M10(x)
        return x1


if __name__ == "__main__":
    example = CNN_doa()
    example = example.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(example.parameters(), lr=learning_rate)

    #   # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 5
    # 添加tensorboard，用于可视化
    writer = SummaryWriter("./logs_seq")

    for i in range(epoch):
        print("-----第{}轮训练开始-----".format(i + 1))
        total_train_step = 0
        # 训练步骤开始
        example.train()
        for data in x_train:
            # 先拆分图像
            position, targets = data
            # 将数据拷贝到device中
            position = position.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            # targets = targets.to(device)
            outputs = example(position)
            loss = loss_fn(outputs, targets)  # 误差的平方和再开方除2，MSEloss计算出来的结果是除过N=540的
            # 给梯度清零
            # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
            optimizer.zero_grad()
            # 反向传播，重新计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            total_train_step = total_train_step + 1
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss5", loss.item(), total_train_step)

        # 只统计每次进行测试的这部分数据总体的正确率，所以不允许计数变量一直累加下去，于是将计数变量在每一轮训练后及每一轮统计之前清零
        # 计数变量
        n = 0
        correct_number = 0
        # 测试开始
        example.eval()
        with torch.no_grad():
            for data in x_test:
                position, targets = data
                position = position.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                outputs = example(position)
                loss = loss_fn(outputs, targets)
                print('模型在测试集上的loss：{}'.format(loss))
                outputs = outputs.cpu().numpy()
                targets = targets.cpu().numpy()
                for k in range(32):
                    n += 1  # 数据的数量从1开始，不可以从0开始，因为n大于等于1时，才有意义。
                    outputs_list = list(outputs[k, :])
                    outputs_max_index = outputs_list.index(max(outputs_list))  # index意思是索引值
                    targets_list = list(targets[k, :])
                    targets_max_index = targets_list.index(max(targets_list))  # index意思是索引值
                    if outputs_max_index == targets_max_index:
                        correct_number += 1
                print('当前正确率：{}%'.format(100 * correct_number / n))
                if i == (epoch - 1):
                    writer.add_scalar("Accuracy", 100 * correct_number / n, n)

        total_test_step = total_test_step + 1
        # 保存每一次训练之后的模型和数据
        if i == (epoch - 1):
            torch.save(example, 'SNR_-10_{}.pth'.format(i))
            print("模型已保存")
        writer.close()
