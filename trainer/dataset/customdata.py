from torch.utils.data import Dataset, DataLoader

import pyarrow.parquet as pq
import torch
import pandas as pd
import numpy as np

import pickle

class CustomDataset(Dataset):

    def __init__(self, img_path, label_path):
        super(CustomDataset, self).__init__()
        with open (img_path, 'rb') as fr:
            self.x_data = pickle.load(fr)
        self.y_data = pd.read_csv(label_path).to_numpy()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_data_list = list(self.x_data)
        img_id = x_data_list[idx]

        x = np.array(self.x_data[img_id], dtype = float)
        x = torch.Tensor(x)

        zeropad = torch.nn.ZeroPad2d((0,0,43,44))
        img = zeropad(x)
        img = img.unsqueeze_(0)

        img = img.repeat(3,1,1)

        target1, target2, target3 = self.y_data[idx][1:4]
        return img, target1, target2, target3
    
if __name__ == '__main__':

    data_path = './data/train_dataset.pickle'
    label_path = './data/train.csv'
    test = CustomDataset(data_path, label_path)

    # dataset = {}
    # dataset['train'], dataset['val'] = torch.utils.data.random_split(test, [160840, 40000], generator=torch.Generator().manual_seed(42))
    # print(len(dataset))
    # print(len(dataset['train']))
    dataloader = DataLoader(test, batch_size = 1, shuffle = True)
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train1, y_train2, y_train3 = samples
        print(y_train1, y_train2, y_train3)
        print(np.shape(x_train))
        