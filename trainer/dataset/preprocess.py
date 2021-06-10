from IPython.display import display
from tqdm import tqdm 

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pickle

def check(data):
    for i in range(len(data)):
        if data[i] <= 250:
            return False
    return True 

dataset = {}
for i in range(4):
    print(f'./data/train_image_data_{i}.parquet')
    tmp = pq.ParquetDataset(f'./data/train_image_data_{i}.parquet')
    tmp = tmp.read()
    tmp = tmp.to_pandas()
    tmp = tmp.to_numpy()

    for j in tqdm(range(len(tmp))):
        img = np.reshape(tmp[j][1:],(137,236))
        cnt = 0 
        while cnt < 12 :
            if check(img[:,0]) and check(img[:, -1]) and cnt != 11:
                img = img[:, 1:]
                img = img[:, :-1]
                cnt += 2
            
            elif check(img[:,0]):
                img = img[:, 1:]
                cnt += 1 
            
            elif  check(img[:, -1]):
                img = img[:, :-1]
                cnt += 1

            elif cnt == 11 and not check(img[:,0]) and not check(img[:, -1]):
                img = img[:,1:]
                cnt += 1

            else :
                img = img[:, 1:]
                img = img[:, :-1]
                cnt += 2

        dataset[tmp[j][0]] = np.array((img), dtype = float)
with open('train_dataset.pickle', 'wb') as fw:
    pickle.dump(dataset, fw)

        
