import torch.nn as nn
num_classes = 2  # 类别数
batch_size = 128  # 批次大小

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(1,3), dilation=(1,2))#！！！不同模型参数不同
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 3),padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 3),padding='same')
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(46080, num_classes)  # ！！！不同模型参数不同

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        print("x.shape: {}".format(x.shape))
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        print("[conv1] x.shape: {}".format(x.shape))
        x = self.pool(x)
        print("[1pool] x.shape: {}".format(x.shape))
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)))
        print("[conv2] x.shape: {}".format(x.shape))
        x = self.pool(x)
        x = nn.functional.leaky_relu(self.bn3(self.conv3(x)))
        print("[conv3] x.shape: {}".format(x.shape))
        x = self.pool(x)
        print("[pool3-pool] x.shape: {}".format(x.shape))
        # x = nn.functional.leaky_relu(self.bn3(self.conv3(x)))
        # x = self.pool(x)
        x = x.view(batch_size, 46080)
        # print("[before flatten] x.shape: {}".format(x.shape))
        x = self.dropout(x)
        # print("[before flatten] x.shape: {}".format(x.shape))
        x = self.fc1(x)
        # x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x