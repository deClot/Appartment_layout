import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.datasets import preprocess_appartment_data, get_dataset
from scripts.options import get_options_from_json
from scripts.get_models import define_G

opt = get_options_from_json()
df = pd.read_csv('/home/refenement/Projects/Dataset_flats/flats_info.csv')


df_prep, ss, ohe = preprocess_appartment_data(df)
train_set = get_dataset(opt['root_path'], df_prep, n_type='21',
                        add_input=opt['add_input'])
test_set = get_dataset(opt['root_path'], df_prep, n_type='21',
                       add_input=opt['add_input'], test=True)
training_data_loader = DataLoader(dataset=train_set, num_workers=4,
                                  batch_size=opt['batch_size'], shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4,
                                 batch_size=opt['batch_size'], shuffle=True)
testing_data_loader_bs1 = DataLoader(dataset=test_set, num_workers=4,
                                     batch_size=1, shuffle=False)
train_close = ['3r_82m28_ts_sy', '3r_82m28_ts_sy',
               '2r_50m48_tt_sy', '2r_42m52_tt_sy',
               '1r_30m1_tt_sn', '1r_40m65_tt_sn']

if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
assert torch.cuda.is_available()
device = torch.device("cuda:0")
g = define_G(3, 3, 'resnet_9blocks', device=device)
