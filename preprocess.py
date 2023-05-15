import pandas as pd
import yaml
import os, sys
import numpy as np
from tqdm import tqdm
from time import sleep
import re
from math import ceil

configPath = 'config.yml'
with open(configPath) as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)


# preprocess_data
def process_data(data_path):
    filename_date = re.search(conf['filename_date_reg'], data_path)[0]
    filename = filename_date+'.csv'
    processed_data_path = os.path.join(conf['processed_data_path'], filename)
    minutely_data_path = os.path.join(conf['minutely_data_path'], filename)
    daily_data_path = os.path.join(conf['daily_data_path'], filename)

    # Save result
    if os.path.exists(processed_data_path):
        os.remove(processed_data_path)
    if os.path.exists(minutely_data_path):
        os.remove(minutely_data_path)
    if os.path.exists(daily_data_path):
        os.remove(daily_data_path)

    chunks = pd.read_csv(data_path,
                         header=None,
                         usecols=[0, 1, 7],
                         chunksize=conf['chunksize'],
                         delimiter='\t'
                         )

    daily_data = pd.DataFrame(
        {
            conf['daily_data_columns'][0]: np.arange(1, 10001),
            conf['daily_data_columns'][1]: np.zeros(10000)
        })

    with open(data_path) as f:
        line_count = len(list(f))
    total = ceil(line_count / conf['chunksize'])

    # 处理每个块
    # with tqdm(chunks, total=100) as pbar:
    pbar = tqdm(chunks, total=total, file=sys.stdout)
    for i, chunk in enumerate(pbar):
        pbar.set_description('Processing %s,%d/%d' % (data_path, i + 1, total))

        # Assign Column Name
        chunk.columns = conf['columns']

        # Fileter NaN
        chunk = chunk.dropna()

        header = i == 0
        chunk.to_csv(processed_data_path, mode='a', index=False, header=header)

        result = chunk.groupby(conf['daily_data_columns'][0])[conf['daily_data_columns'][1]].sum()

        # for i in result:
        daily_data.loc[
            result.index - 1,
            conf['daily_data_columns'][1]
        ] += result.values

    # daily
    daily_data.to_csv(daily_data_path, index=False)

    # minutely
    processed_data = pd.read_csv(processed_data_path)
    result = processed_data.groupby([conf['columns'][0],conf['columns'][1]])[conf['columns'][2]].sum()
    result.reset_index().to_csv(minutely_data_path, index=False)
# ==============================
if __name__ == '__main__':

    files = os.listdir(conf['raw_file_path'])
    files = filter(lambda filename: re.search(conf['filename_date_reg'], filename), files)

    for i in files:
        filepath = os.path.join(conf['raw_file_path'], i)
        process_data(filepath)

