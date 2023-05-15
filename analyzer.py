import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
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


def getData():
    traffic = pd.DataFrame(
        {
            CELLID: np.arange(1, 10001),
            TRAFFIC: np.zeros(10000)
        })

    files = list(Path(conf['daily_data_path']).iterdir())
    files.sort()


    start = re.search(conf['filename_date_reg'], str(files[0]))[0]
    end = re.search(conf['filename_date_reg'], str(files[-1]))[0]
    for i in files:
        # print(i)
        df = pd.read_csv(i)

        traffic.loc[
            df[CELLID] - 1,
            TRAFFIC
        ] += df[TRAFFIC].values

    return traffic, [start,end]

def getBasicInfo(df):
    maxV = max(df[TRAFFIC])
    minV = min(df[TRAFFIC])
    mean = df[TRAFFIC].mean()
    std = df[TRAFFIC].std()

    uq = np.quantile(df[TRAFFIC], 0.25)
    median = np.quantile(df[TRAFFIC], 0.5)
    lq = np.quantile(df[TRAFFIC], 0.75)
    print(maxV,minV,mean,std)
    print(uq, median, lq)
    return {'maxV':maxV, 'minV':minV, 'mean':mean, 'std':std, 'uq':uq, 'median':median, 'lq':lq}

def getHistogram(df, startEndRange):

    # Create histogram
    # Set density=True to convert histogram to PDF plot
    # Set alpha=0.5 to make the histogram semi-transparent



    df_sorted = df.sort_values(by=TRAFFIC, ascending = False)

    # 取前10%的数据
    num_rows = len(df_sorted)
    df_top_10_percent = df_sorted.head(int(num_rows * 0.1))
    df_top_5_percent = df_sorted.head(int(num_rows * 0.05))

    display(df, '100 percent', startEndRange)
    display(df_top_10_percent, 'top 10 percent', startEndRange)
    display(df_top_5_percent, 'top 5 percent', startEndRange)


def display(df, name, startEndRange):
    filepath = os.path.join(conf['analysis_path'], name + '.pdf')

    n, bins, patches = plt.hist(df[TRAFFIC], bins=20, density=True, alpha=0.5)

    mean = df[TRAFFIC].mean()
    std = df[TRAFFIC].std()

    # Calculate probability density function (PDF)
    pdf = 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-(bins - mean) ** 2 / (2 * std ** 2))
    # Plot the PDF as a line plot
    # Set linewidth=2 and color='r' to make the line thicker and red
    plt.plot(bins, pdf, linewidth=2, color='r')

    # Add labels and title to the plot
    plt.xlabel('TRAFFIC')
    plt.ylabel('Frequency')
    plt.title(f'Probability Density Function of TRAFFIC of %s\nfrom %s to %s'%(name, startEndRange[0], startEndRange[1]))

    # Display the plot
    # plt.show()
    plt.savefig(filepath)
    plt.close()

def getEntropy(df):

    bin_list = conf['bin_list']
    entropyList = []
    maxList = []
    for bins in bin_list:
        # 将数据划分成20个区间
        hist, bin_edges = np.histogram(df[TRAFFIC], bins=bins, density=True)
        # 计算每个区间的概率
        p = hist * np.diff(bin_edges)
        # 计算分布的熵
        ent = entropy(p, base=2)
        # 输出熵的值
        entropyList.append(ent)
        maxList.append(np.log2(bins))
        # print("Entropy of the distribution:", ent)

    for x, y in zip(bin_list, entropyList):
        plt.text(x + 0.05, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    plt.bar(bin_list, entropyList, label='entropy')
    plt.plot(bin_list, maxList, label='maxEntropy')
    plt.xlabel('# Interval')
    plt.ylabel('Entropy')
    plt.legend()
    plt.savefig(conf['entropy_fullpath'])
    plt.show()

def getDistribution(df, basicInfo, startEndRange):
    n, bins, patches = plt.hist(df[TRAFFIC], bins=20, density=False, alpha=0.5)
    print(basicInfo)
    plt.axvline(x=basicInfo['lq'], ls="--", lw=2, label='lq')
    plt.axvline(x=basicInfo['median'], ls="--", lw=2, label='median')
    plt.axvline(x=basicInfo['uq'], ls="--", lw=2, label='uq')
    # Add labels and title to the plot
    plt.xlabel('TRAFFIC')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Grid Cell\nfrom %s to %s'%(startEndRange[0], startEndRange[1]))
    # 添加图例
    plt.legend()

    # Display the plot
    plt.savefig(conf['distribution_fullpath'])
    plt.show()
    plt.close()

def getFirstTwoWeeksFlow():

    files = list(Path(conf['minutely_data_path']).iterdir())
    files.sort()
    files=files[0:14]

    res = {i:[] for i in conf['cellIdList']}
    for f in files:
        df = pd.read_csv(f)
        for id in conf['cellIdList']:

            res[id].extend(df[df[CELLID]==id][TRAFFIC].values)

    fig = plt.figure(figsize=(12, 4))
    for id in conf['cellIdList']:
        plt.plot(res[id], label = id)
    plt.legend()
    plt.savefig(conf['FirstTwoWeeksFlow_fullpath'])
    plt.show()

if __name__ == '__main__':

    df, startEndRange = getData()

    # basicInfo = getBasicInfo(df)
    # getHistogram(df,startEndRange)
    # getDistribution(df, basicInfo, startEndRange)
    # getEntropy(df)
    getFirstTwoWeeksFlow()

