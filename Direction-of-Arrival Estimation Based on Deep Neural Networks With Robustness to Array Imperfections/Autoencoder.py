import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from dataset_autoencoder import Data_Loader

# 定义参与训练的设备（cpu or gpu）
if __name__ == "__main__":
    device = torch.device("cuda")

# 准备数据集
if __name__ == "__main__":
    data_dataset = Data_Loader("./自编码器/SNR20/Autoencoder.mat")
    train_dataloader = torch.utils.data.DataLoader(dataset=data_dataset, shuffle=True, batch_size=32, drop_last=False)


class DNN_Autoencoder(nn.Module):
    def __init__(self, input_length, num_layers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_length, 45, bias=True),
            nn.Tanh(),
            nn.Linear(45, 22, bias=True),
            nn.Tanh(),
        )
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(22, 45, bias=True),
                nn.Tanh(),
                nn.Linear(45, input_length, bias=True),
                nn.Sigmoid())
            self.add_module("deconder%d" % (i + 1), layer)

        # self.layer = nn.Linear(22, 540, bias=True)

    # def forward(self, x):
    #     # x = self.conv(x)
    #     x1 = self.encoder(x)
    #     x2 = self.layer(x1)
    #
    #     return x2
    def forward(self, x):
        # x = self.conv(x)
        x1 = self.encoder(x)
        x2 = self.deconder1(x1)
        x3 = self.deconder2(x1)
        x4 = self.deconder3(x1)
        x5 = self.deconder4(x1)
        x6 = self.deconder5(x1)
        x7 = self.deconder6(x1)
        return torch.cat([x2, x3, x4, x5, x6, x7], dim=2)


if __name__ == "__main__":
    # 创建模型
    example = DNN_Autoencoder(90, 6)
    example = example.to(device)

    # 损失函数
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)

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
    epoch = 5000
    # 添加tensorboard，用于可视化
    writer = SummaryWriter("./logs_seq")
    # 计数变量
    n = 0

    for i in range(epoch):
        print("-----第{}轮训练开始-----".format(i + 1))
        total_train_step = 0
        # 训练步骤开始
        example.train()
        for data in train_dataloader:
            # 先拆分图像
            imgs, targets = data
            # 将数据拷贝到device中
            imgs = imgs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            # targets = targets.to(device)
            outputs = example(imgs)
            loss = 540 * 0.5 * loss_fn(outputs, targets)  # 误差的平方和再开方除2，MSEloss计算出来的结果是除过N=540的
            # 给梯度清零
            # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 重新计算梯度，更新参数
            optimizer.step()
            total_train_step = total_train_step + 1
            if total_train_step == 3:
                print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss5", loss.item(), total_train_step)

        if i == epoch - 1:
            torch.save(example, 'Autoencoder_20{}.pth'.format(i))
            print("模型已保存")
        # writer.close()
