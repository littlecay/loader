import torch
import numpy as np
from torch.utils.data import Dataset        # 构造数据集 支持索引，总长度
from torch.utils.data import DataLoader     # 拿mini-batch

"""
    1. Prepare dataset
        tools: Dataset and DataLoader
        
    2. Design model using Class
        inherit from nn.Module
        
    3. Construct loss and optimizer
        using PyTorch API
        
    4. Training cycle
        forward, backward, update
"""


class MyDataset(Dataset):
    def __init__(self, filepath):
        """
        处理数据的两种做法：
            1: All Data load to Memory  （结构化数据）
            2: 定义一个列表，把每个 sample 路径 放到一个列表，标签放到另一个列表里，避免数据一次性全部加入到内存 （非结构化数据）
        """
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        print("Data ready")

    def __getitem__(self, index):    # 为了支持下标操作，即索引 dataset[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self):              # 为了使用 len(dataset)
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


file = "./dataset/diabetes/diabetes.csv"

""" 1.使用 MyDataset类 构建自己的dataset """
mydataset = MyDataset(file)
""" 2.使用 DataLoader 构建train_loader """
train_loader = DataLoader(dataset=mydataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

model = Model()

criterion = torch.nn.BCELoss(size_average=True)             # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)     # 优化器优化参数

if __name__ == "__main__":
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1.准备数据
            inputs, labels = data   # Tensor type

            # 2.前向传播
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            # 3.后向传播
            optimizer.zero_grad()
            loss.backward()

            # 4.更新
            optimizer.step()
