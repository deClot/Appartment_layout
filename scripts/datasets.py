from PIL import Image
from os.path import join
import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, image_filenames, n_type,
                 df, add_input=True, img_fmt='png'):
        super(DatasetFromFolder, self).__init__()
        # self.direction = direction
        self.image_dir = image_dir
        self.a_path = '_256_1' + '.' + img_fmt  # join(image_dir, "a")
        self.b_path = '_256_' + n_type + '.' + img_fmt  # join(image_dir, "b")
        self.image_filenames = image_filenames
        self.data = df
        self.add_input = add_input

    def __getitem__(self, index):
        a = Image.open(join(self.image_dir,
                            self.image_filenames[index]+self.a_path))\
                 .convert('RGB')
        b = Image.open(join(self.image_dir,
                            self.image_filenames[index]+self.b_path))\
                 .convert('RGB')
        a, b, appartment_info_img, name = \
            transformation_sample(a, b, self, index=index)
        return a, b, appartment_info_img, name

    def __len__(self):
        return len(self.image_filenames)


def transformation_sample(a, b, dataset, index=None, name=None):
    a = transforms.ToTensor()(a)
    b = transforms.ToTensor()(b)
    a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
    b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

    # reverse img
    if random.random() < 0.5:
        idx = [i for i in range(a.size(2)-1, -1, -1)]  # revese img
        idx = torch.LongTensor(idx)
        a = a.index_select(2, idx)
        b = b.index_select(2, idx)

    name = dataset.image_filenames[index] if index is not None else name
    if dataset.add_input:
        appartment_info = dataset.data[dataset.data.name == name]\
                        .drop(['name', 'test'], axis=1).values.reshape(-1, 1)
        appartment_info = appartment_info[:, :, None].astype(np.float)
        appartment_info_img = torch.Tensor(appartment_info)\
                                   .repeat(1, a.shape[1], a.shape[2])
    else:
        appartment_info_img = torch.Tensor()

    return a, b, appartment_info_img, name


def get_sample_by_name(dataset, name):
    a = Image.open(join(dataset.image_dir, name+dataset.a_path)).convert('RGB')
    b = Image.open(join(dataset.image_dir, name+dataset.b_path)).convert('RGB')
    a, b, appartment_info_img, name = transformation_sample(a, b, dataset,
                                                            name=name)
    return a, b, appartment_info_img, name


def preprocess_appartment_data(df):
    train = df[df.test != True]
    to_ohe = ['n_rooms', 'restroom', 'studio', 'storeroom']
    ss = StandardScaler().fit(train[['area']])
    ohe = OneHotEncoder(drop='first', sparse=False).fit(train[to_ohe])

    df_ss = ss.transform(df[['area']])
    df_ohe = ohe.transform(df[to_ohe])
    df = pd.DataFrame(np.hstack([df[['name', 'test']], df_ss, df_ohe]),
                      columns=['name', 'test', 'area', 'n_rooms2', 'n_rooms3',
                               'restroom_separate', 'restroom_together',
                               'studio', 'storeroom'])
    return df, ss, ohe


def get_dataset(root_dir, df, n_type, add_input, test=False):
    df = df[df.test == True] if test else df[df.test == False]
    filenames = df.name.tolist()
    size_info = 'Test' if test else 'Train'
    size_info += f' size:\t{len(filenames):5d}'
    print(size_info)
    return DatasetFromFolder(root_dir, filenames, n_type, df,
                             add_input=add_input)


def postproces_img(img_tensor):
    img = img_tensor.data.cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1.) / 2. * 255.
    return img.astype(np.uint8)
