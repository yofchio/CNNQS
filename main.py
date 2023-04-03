import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split

from torch import optim

from early_stopping import EarlyStopping
from CNNmodel import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
width = 60  # 图像宽度
height = 64  # 图像高度
num_epochs = 20  # 迭代次数
batch_size = 128  # 批次大小

#早停止参数
save_path = ".\\" #当前目录下
early_stopping = EarlyStopping(save_path)
#定义训练参数
train_loss = 0
train_acc = 0
train_losses = []
train_acces = []
state_dict_PATH="best_network.pth" #保存的最好模型参数字典
load_static_dict=False# 是否加载字典，默认是false，如果出现代码突然中止，以及需要val的情况，那么设置为True
# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
eval_losses = []
eval_acces = []
val_name=[]
val_rote=[]
# 读取CSV数据并生成2D矩阵
data_train = pd.read_csv("data/data_20_1993-2000.csv")
data_val = pd.read_csv("data/data_20_2001-2019.csv")
X_T = np.array(data_train.iloc[:, 2:]) # 训练的特征矩阵
y_T = np.array(data_train.iloc[:, 1])  # 训练的标签
X_V = np.array(data_val.iloc[:, 2:]) # 验证的特征矩阵
y_V = np.array(data_val.iloc[:, 1])  # 验证的标签
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_T, y_T, test_size=0.3, random_state=42)

# 创建训练数据加载模型
class CsvtrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CsvtrainDataset, self).__init__()

        self.feature = X_train

        self.label = y_train

        assert len(self.feature) == len(self.label)

        self.length = len(self.feature)

    def __getitem__(self, index):
        x = self.feature[index]
        x = torch.Tensor(x)
        x = x.reshape(1, height, width)

        y = self.label[index]

        return x, y

    def __len__(self):
        return self.length

# 创建测试数据加载模型
class CsvtestDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CsvtestDataset, self).__init__()

        self.feature = X_test

        self.label = y_test

        assert len(self.feature) == len(self.label)

        self.length = len(self.feature)

    def __getitem__(self, index):
        x = self.feature[index]
        x = torch.Tensor(x)
        x = x.reshape(1, height, width)
        y = self.label[index]
        return x, y
    def __len__(self):
        return self.length

# 创建验证数据加载模型
class CsvvalDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(CsvvalDataset, self).__init__()

        self.feature = X_V

        self.label = y_V

        assert len(self.feature) == len(self.label)

        self.length = len(self.feature)

    def __getitem__(self, index):
        x = self.feature[index]
        x = torch.Tensor(x)
        x = x.reshape(1, height, width)
        y = self.label[index]
        return x, y
    def __len__(self):
        return self.length

train_dataset = CsvtrainDataset()
test_dataset = CsvtestDataset()
val_dataset = CsvvalDataset()
# 将数据集加载到数据加载器中
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

# 初始化模型和损失函数
model = CNN().to(device)
if load_static_dict:
    if state_dict_PATH is not None:
        model.load_state_dict(torch.load(state_dict_PATH))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
        print(f'resume model from {state_dict_PATH}')
    else:
        print('No model found, initializing random model.')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.00001)

def train(epoch):
    train_loss = 0
    train_acc = 0
    for batch_idx, data in enumerate(trainloader, 0):   #在这里data返回输入:inputs、输出target
        inputs, target = data
        # print(inputs)
        # print('\n')
        # print(target)
        # print('\n')
        #在这里加入一行代码，将数据送入GPU中计算！！！
        inputs, target = inputs.to(device), target.to(device)
        # 梯度置为零
        optimizer.zero_grad()

        #前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        out_t = outputs.argmax(dim=1)  # 取出预测的最大值
        train_loss += loss.item()
        num_correct = (out_t == target).sum().item()
        acc = num_correct / inputs.shape[0]
        train_acc += acc

    train_losses.append(train_loss / len(trainloader))
    train_acces.append(train_acc / len(trainloader))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'
          .format(epoch, train_loss / len(trainloader), train_acc / len(trainloader)))

def test(epoch):
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():  #不需要计算梯度
        for data in testloader:   #遍历数据集中的每一个batch
            images, labels = data  #保存测试的输入和输出
            #在这里加入一行代码将数据送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)#得到预测输出
            loss = criterion(outputs, labels)  # 得到误差
            eval_loss += loss.item()

            # 记录准确率
            out_t = outputs.argmax(dim=1)  # 取出预测的最大值的索引
            num_correct = (out_t == labels).sum().item()  # 判断是否预测正确
            acc = num_correct / images.shape[0]  # 计算准确率
            eval_acc += acc
        eval_losses.append(eval_loss / len(testloader))
        eval_acces.append(eval_acc / len(testloader))
        print('epoch: {}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(epoch, eval_loss / len(testloader), eval_acc / len(testloader)))
        # 早停止
    early_stopping(eval_loss, model)
    # 达到早停止条件时，early_stop会被置为True
    if early_stopping.early_stop:
        print("Early stopping")
        model.eval()
        val()
        sys.exit(1)


def val():
    correct = 0
    total = 0
    with torch.no_grad():  #不需要计算梯度
        print(valloader)
        for data in valloader:   #遍历数据集中的每一个batch
            images, labels = data  #保存测试的输入和输出
            #在这里加入一行代码将数据送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)#得到预测输出
            # print(outputs.data.size())
            _, predicted = torch.max(outputs.data, dim=1)#dim=1沿着索引为1的维度(行)
            # print(_)
            val_rote.extend( _.numpy().tolist())
            # print(val_rote)
            val_name.extend(predicted.numpy().tolist())
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))
    dataframe = pd.DataFrame({'return': val_name, 'rate': val_rote})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("return-rate-20.csv", index=True, sep=',')

if __name__ == '__main__':

    for epoch in range(num_epochs):
        train(epoch)
        # val()
        test(epoch)
