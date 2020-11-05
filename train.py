import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import neptune

from scripts.datasets import preprocess_appartment_data, get_dataset
from scripts.options import get_options_from_json
from scripts.get_models import define_G, define_D
from scripts.train import GANLoss, get_scheduler, check_label
from scripts.train import set_requires_grad, update_learning_rate
from scripts.train import get_generator_metric, get_gan_metric, save_test_metrics
from scripts.train import create_checkpoint, plot_pred, save_roc
from scripts.train import get_test_predictions


batch_size = 0


def train(datas, n_epochs=200, lr=.0002, beta1=0.5, lambda_l1=10, device=None,
          epoch_start=1, n_epochs_decay=10,
          lr_decay_iters=10, lr_policy='linear',
          netD='basic', n_layers=3,
          use_dropout=True,
          verbose=2, name='', to_neptune=True):

    training_data_loader, testing_data_loader, testing_data_loader_bs1 = datas
    add_nc = training_data_loader.dataset[0][2].shape[0]
    print(f'# of channel of additional input vector: {add_nc}')
    add_input = True if add_nc > 0 else False
    net_g = define_G(3 + add_nc, 3, 'resnet_9blocks', 'batch', use_dropout,
                     'normal', 0.02, device=device)
    net_d = define_D(6 + add_nc, 64, netD, n_layers=n_layers, gpu_id=device)
    patches = net_d(torch.zeros((1, 6+add_nc, 256, 256)).to(device)).shape[-2]
    print(f'# of patche for D: {patches}x{patches}')

    if to_neptune:
        neptune.create_experiment(name=name,
                                  params={'batch_size': batch_size,
                                          'lr': lr,
                                          'lambda_l1': lambda_l1,
                                          'n_epochs': n_epochs,
                                          'd_patches': patches,
                                          'use_dropout': use_dropout},
                                  upload_source_files=['train.py',
                                                       'params.json',
                                                       'scripts/'])

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    # criterionMSE = nn.MSELoss().to(device)

    # setup optimizer
    epochs_info = epoch_start, n_epochs, n_epochs_decay, lr_decay_iters
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, lr_policy, epochs_info)
    net_d_scheduler = get_scheduler(optimizer_d, lr_policy, epochs_info)

    for epoch in tqdm(range(epoch_start, n_epochs + n_epochs_decay + 1)):
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b, labels = batch[0].to(device),\
                batch[1].to(device),\
                check_label(batch[2].to(device))
            real_a_labels = torch.cat((real_a, labels), 1)
            fake_b = net_g(real_a_labels)
            torch.autograd.set_detect_anomaly(True)
            ######################
            # (1) Update D network
            ######################

            set_requires_grad(net_d, True)
            optimizer_d.zero_grad()

            # forward D with fake, detach G
            fake_ab = torch.cat((real_a, labels, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # forward D with real
            real_ab = torch.cat((real_a, labels, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)

            # Combined D loss and backward
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            set_requires_grad(net_d, False)
            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            # forward G fake as 1
            fake_ab = torch.cat((real_a, labels, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * lambda_l1

            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            optimizer_g.step()
            if to_neptune:
                neptune.log_metric('g', loss_g.data.cpu().numpy())
                neptune.log_metric('g_gan', loss_g_gan.data.cpu().numpy())
                neptune.log_metric('g_l1', loss_g_l1.data.cpu().numpy())
                neptune.log_metric('d', loss_d.data.cpu().numpy())
                neptune.log_metric('d0', loss_d_fake.data.cpu().numpy())
                neptune.log_metric('d1', loss_d_real.data.cpu().numpy())

        update_learning_rate(net_g_scheduler, optimizer_g, epoch, show=to_neptune)
        update_learning_rate(net_d_scheduler, optimizer_d, epoch, show=False)

        if to_neptune:
            neptune.log_metric('psnr', (epoch-1)*len(training_data_loader),
                               get_generator_metric(training_data_loader,
                                                    net_g, device))
            save_test_metrics(testing_data_loader, len(training_data_loader),
                              epoch, net_g, net_d, lambda_l1, device)

        if epoch % verbose == 0 and to_neptune:
            create_checkpoint(epoch, net_g, net_d)
            plot_pred(training_data_loader, testing_data_loader,
                      net_g, device, epoch)
            save_roc(training_data_loader, epoch, net_g, net_d,
                     len(training_data_loader), device)
            save_roc(testing_data_loader, epoch, net_g, net_d,
                     len(training_data_loader), device, test=True)
    if to_neptune:
        all_pred = get_test_predictions(epoch, testing_data_loader_bs1,
                                        train_set, train_close,
                                        device, add_input)
        neptune.log_image('pred_all', all_pred)
        neptune.stop()


if __name__ == '__main__':
    neptune.init('declot/alpix2pix')

    opt = get_options_from_json()
    df = pd.read_csv('/home/refenement/Projects/Dataset_flats/flats_info.csv')
    df_ini = pd.read_csv('/home/refenement/Projects/Dataset_flats/flats_info_ini.csv')
    print(df_ini.shape)
    df_prep, ss, ohe = preprocess_appartment_data(df)
    train_set = get_dataset(opt['root_path'], df_prep, n_type='21',
                            add_input=opt['add_input'])
    test_set = get_dataset(opt['root_path'], df_prep, n_type='21',
                           add_input=opt['add_input'], test=True)
    test_set_ini = get_dataset(opt['root_path'], df_prep.iloc[:df_ini.shape[0]],
                               n_type='21', add_input=opt['add_input'], test=True)
    training_data_loader = DataLoader(dataset=train_set, num_workers=4,
                                      batch_size=opt['batch_size'], shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=4,
                                     batch_size=opt['batch_size'], shuffle=True)
    testing_data_loader_bs1 = DataLoader(dataset=test_set_ini, num_workers=4,
                                         batch_size=1, shuffle=False)
    train_close = ['3r_82m28_ts_sy', '3r_82m28_ts_sy',
                   '2r_50m48_tt_sy', '2r_42m52_tt_sy',
                   '1r_30m1_tt_sn', '1r_40m65_tt_sn']

    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    train((training_data_loader, testing_data_loader, testing_data_loader_bs1),
          n_epochs=opt['n_epochs'], lambda_l1=opt['lambda_l1'],
          netD=opt['net_D'], n_layers=opt['n_layers'],
          use_dropout=opt['use_dropout'],
          verbose=opt['verbose'], n_epochs_decay=0, name=opt['name'],
          device=device, to_neptune=opt['to_neptune'])

    # # get_test_predictions(100, testing_data_loader_bs1, train_set, train_close, device)
    # from scripts.models_hd import LocalEnhancer

    # LocalEnhancer(3,3)
