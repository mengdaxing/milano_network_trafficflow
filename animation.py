import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os
import re
import imageio.v3 as iio
from pathlib import Path
import datetime
from datetime import timezone, timedelta

configPath = 'config.yml'
with open(configPath) as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)


CELLID = conf['columns'][0]
TIMESTAMP = conf['columns'][1]
TRAFFIC = conf['columns'][2]

def display(ts, df):
    frame_data_path = os.path.join(conf['animation_frame_path'], str(ts)+'.png')
    traffic = pd.DataFrame(
        {
            CELLID: np.arange(1, 10001),
            TRAFFIC: np.zeros(10000)
        })
    traffic.loc[
        df[CELLID] - 1,
        TRAFFIC
    ] = df[TRAFFIC].values

        # Minutely Result
    traffic = traffic.loc[:, TRAFFIC].values.reshape(100,100)

    # 绘制图像
    plt.imshow(traffic, cmap='Blues')

    # 绘制概率密度函数等高线图
    x = np.linspace(0, 99, 100)
    y = np.linspace(0, 99, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X-50)**2 + (Y-50)**2) / 200)
    # plt.contour(X, Y, Z, colors='red')
    title = datetime.datetime.fromtimestamp(int(ts/1000))\
        .astimezone(timezone(timedelta(hours=1)))
    plt.title(title)
    # 显示图形
    # plt.show()
    plt.savefig(frame_data_path)

def getFrame(filepath):
    # 生成 100x100 的随机数据
    df = pd.read_csv(filepath)

    for ts, frame in df.groupby([TIMESTAMP]):
        display(ts, frame)
def getAnimation():

    images = list()
    files = list(Path(conf['animation_frame_path']).iterdir())
    files.sort()
    for file in files:
        if not (file.is_file() and file.suffix == '.png'):
            continue

        images.append(iio.imread(file))

    iio.imwrite(conf['animation_final_fullpath'], images, duration=100)


if __name__ == '__main__':

    files = os.listdir(conf['minutely_data_path'])
    res = filter(lambda filename: re.match(conf['filename_date_reg'], filename), files)
    for i in (list(res)):
        filepath = os.path.join(conf['minutely_data_path'], i)
        getFrame(filepath)
        getAnimation()
        break
