import torch
import torch.nn as nn
import numpy as np
from model_back import TCN
import pandas as pd

from pathlib import Path
import yaml
import re
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

configPath = 'config.yml'
with open(configPath) as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

CELLID = conf['columns'][0]
TIMESTAMP = conf['columns'][1]
TRAFFIC = conf['columns'][2]

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
else:
    device = torch.device("cpu")  # 如果没有 GPU，使用 CPU

# 定义 scaler
scaler = StandardScaler()

# 定义数据预处理函数，将时间序列数据转换为窗口特征
def getDataset(window_size):

    df = pd.read_csv(conf['the_three_area_data_fullpath'])
    # 准备时间序列数据
    data = df[TRAFFIC].values
    data = data.reshape(-1, 3)
    data = scaler.fit_transform(data)


    trainX = []
    trainY = []
    testX = []
    testY = []


    for i in range(len(data) - window_size - 2*conf['outputSize'] + 1):
        trainX.append(data[i: i+window_size])
        trainY.append(data[i+window_size: i+window_size+conf['outputSize']])

    testX.append(data[len(data) - window_size - conf['outputSize']: len(data) - conf['outputSize']])
    testY.append(data[len(data) - conf['outputSize']: len(data)+1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    # 将数据转换为PyTorch张量
    trainX = torch.Tensor(trainX).transpose(1,2).to(device)
    trainY = torch.Tensor(trainY).transpose(1,2).to(device)
    testX = torch.Tensor(testX).transpose(1,2).to(device)
    testY = torch.Tensor(testY).transpose(1,2).to(device)

    # 将数据划分为训练集和测试集
    train_dataset = torch.utils.data.TensorDataset(trainX, trainY)
    test_dataset = torch.utils.data.TensorDataset(testX, testY)

    return train_dataset, test_dataset

def train(windowSize):
    train_dataset, test_dataset = getDataset(windowSize)
    batch_size = conf['batchSize']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建TCN模型和优化器
    input_size = train_dataset[0][0].shape[1]
    output_size = train_dataset[0][1].shape[1]
    num_channels = conf['num_channels']
    output_channels = train_dataset[0][1].shape[0]
    kernel_size = conf['kernel_size']
    dropout = 0.2
    model = TCN(input_size, output_size, num_channels, kernel_size, dropout, output_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])
    summary(model,
            input_size=(train_dataset[0][0].shape[0], train_dataset[0][0].shape[1]),
            device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 训练TCN模型
    for epoch in range(conf['num_epochs']):
        for i, (inputs, targets) in enumerate(train_loader):
            # 将数据传递到模型中进行训练
            outputs = model(inputs)

            loss = nn.functional.mse_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出训练信息
            # if (i+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{conf["num_epochs"]}], Loss: {loss.item():.4f}')

        torch.save(model, conf['model_fullpath'])

def getTheThreeData():

    files = list(Path(conf['minutely_data_path']).iterdir())
    files.sort()

    total = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f)
        total = pd.concat(
            [total,
            df[df[CELLID].isin(conf['cellIdList'])]]
        )
        if re.search(conf['test_end_date'], str(f)):
            break


    total = total.sort_values(by=[TIMESTAMP, CELLID], ascending=True)\
        .reindex(columns=[TIMESTAMP, CELLID, TRAFFIC])\
        .reset_index(drop=True)
    total.to_csv(conf['the_three_area_data_fullpath'], index=False)

def predict(windowSize):
    model = torch.load(conf['model_fullpath']).to(device)
    train_dataset, test_dataset = getDataset(windowSize)
    batch_size = conf['batchSize']
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 在测试集上评估TCN模型
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader: # batch, channel, traffic
            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)

            rmse = np.sqrt(loss.item())
            print(f'Test RMSE: {rmse:.4f}')

            outputs = outputs.to('cpu')
            targets = targets.to('cpu')
            outputs = outputs.T.squeeze()
            targets = targets.T.squeeze()

            outputs = scaler.inverse_transform(outputs)
            targets = scaler.inverse_transform(targets)

            idlist=sorted(conf['cellIdList'])
            fig, axs = plt.subplots(3, figsize=(12, 6))
            for i in range(len(idlist)):
                y = targets[:, i].squeeze()
                y_pred = outputs[:, i].squeeze()

                axs[i].plot(y, label=f'Truth %s'%(idlist[i]))
                axs[i].plot(y_pred, label=f'Prediction %s'%(idlist[i]))
                axs[i].set_title(idlist[i])
            # plt.legend()
            fig.suptitle(f'Prediction and Truth Value (input_len = %s, loss={rmse:.4f})'
                         %(windowSize)
            )
            fig.savefig(conf['pred_result_fullpath']%(windowSize))
            fig.show()



if __name__ == '__main__':

    # getTheThreeData()
    for windowSize in conf['windowSize']:
        train(windowSize)
        predict(windowSize)

