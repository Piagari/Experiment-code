import torch
from Autoencoder import DNN_Autoencoder

# 调用方式1，对应的是保存方式1，调用出来是一个未被激活的模块的集合，需要进行赋值激活一下
model = torch.load("example_9999.pth")
print(model)

# 调用方式2，对应的是保存方式2，加载已经写好的模型，然后调用已经保存好的参数，若调用的模型有参数则会替换掉之前的参数
vgg16 = DNN_Autoencoder(90, 6)
vgg16.load_state_dict(torch.load("example_9999.pth"))
print(vgg16)

#用的时候需要import原有的模型 调
