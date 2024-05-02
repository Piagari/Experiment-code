import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from dataset_classifier import Data_Loader
from Autoencoder import DNN_Autoencoder
if __name__ == "__main__":
    device = torch.device("cuda")
    encoder = torch.load("example_99999.pth")

# 准备数据集
if __name__ == "__main__":
    data_dataset = Data_Loader('./自编码器/SNR20/Classifier.mat')
    train_dataloader = torch.utils.data.DataLoader(dataset=data_dataset, shuffle=False, batch_size=32, drop_last=False)

partitions = 90  # 向量的长度除以6，得到每个部分的大小
class DNN_multilayer(nn.Module):
    def __init__(self, input_length, num_layers):
        super(DNN_multilayer, self).__init__()
        layer = Sequential(nn.Linear(input_length, 30),  # 每经过一层，通道数都要增加
                           nn.Tanh(),
                           nn.Linear(30, 20),
                           # nn.Sigmoid()
                           nn.Tanh()
                           )
        for a in range(num_layers):  # 从0开始 0 1
            self.add_module("multilayer%d" % (a + 1), layer)

    def forward(self, x):
        # 使用切片操作符将向量分成六个部分，并使用解包语法赋值给六个变量
        part1 = x[:, 0:partitions]
        part2 = x[:, partitions:2 * partitions]
        part3 = x[:, 2 * partitions:3 * partitions]
        part4 = x[:, 3 * partitions:4 * partitions]
        part5 = x[:, 4 * partitions:5 * partitions]
        part6 = x[:, 5 * partitions:6 * partitions]
        x1 = self.multilayer1(part1)
        x2 = self.multilayer2(part2)
        x3 = self.multilayer3(part3)
        x4 = self.multilayer4(part4)
        x5 = self.multilayer5(part5)
        x6 = self.multilayer6(part6)
        return torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

if __name__ == "__main__":
    # 创建模型
    example = DNN_multilayer(90, 6)
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
    epoch = 300
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
            reg = encoder(imgs)
            outputs = example(reg)
            loss = 60*loss_fn(outputs, targets)  # 误差的平方和再开方除2，MSEloss计算出来的结果是除过N=120的
            # 给梯度清零
            # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 重新计算梯度，更新参数
            optimizer.step()
            total_train_step = total_train_step + 1
            # print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
        if i == 299:
            torch.save(example, 'example_classifier20{}.pth'.format(i+1))
            print("模型已保存")