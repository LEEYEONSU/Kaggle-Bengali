from torch.utils.data import Dataset, DataLoader

import pyarrow.parquet as pq
import torch
import pandas as pd
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, img_path, label_path):
        super(CustomDataset, self).__init__()
        self.x_data = np.load(img_path, allow_pickle = True)
        self.y_data = pd.read_csv(label_path).to_numpy()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        img_id = self.x_data[idx][0]
        x = np.array(self.x_data[idx][1:], dtype = float)
        x = torch.Tensor(x)
        img = torch.reshape(x, (137, 236))

        target = self.y_data[idx][1:4]
        target = np.array(target, dtype = float)
        target = torch.Tensor(target)

        return img, target
    
# data_path = './data/train_image_data_0.parquet'
data_path = './data/train_data0.npy'
label_path = './data/train.csv'
test = CustomDataset(data_path, label_path)
dataloader = DataLoader(test, batch_size = 2, shuffle = True)

for batch_idx, samples in enumerate(dataloader):
    x_train, y_train = samples
    print(samples)



    