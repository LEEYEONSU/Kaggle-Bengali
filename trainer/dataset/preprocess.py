from IPython.display import display
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


# Merge data to one file 
p0 = pq.ParquetDataset('./data/bengaliai-cv19/train_image_data_0.parquet')
p1 = pq.ParquetDataset('./data/bengaliai-cv19/train_image_data_1.parquet')
p2 = pq.ParquetDataset('./data/bengaliai-cv19/train_image_data_2.parquet')
p3 = pq.ParquetDataset('./data/bengaliai-cv19/train_image_data_3.parquet')

table0 = p0.read()
table1 = p1.read()
table2 = p2.read()
table3 = p3.read()

t0 = table0.to_pandas()
t1 = table1.to_pandas()
t2 = table2.to_pandas()
t3 = table3.to_pandas()

t0 = t0.to_numpy()
t1 = t1.to_numpy()
t2 = t2.to_numpy()
t3 = t3.to_numpy()

dataset =[]
dataset.extend(t0)
dataset.extend(t1)
dataset.extend(t2)
dataset.extend(t3)

np.save('./data/train_dataset.npy', dataset)