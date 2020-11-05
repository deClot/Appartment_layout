import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import neptune

from scripts.datasets import preprocess_appartment_data, get_dataset
from scripts.options import get_options_from_json
from scripts.get_models import define_G_hd, define_D_hd
from scripts.train import GANLoss_hd, VGGLoss, check_label

from scripts.train import get_generator_metric
from scripts.train import create_checkpoint, plot_pred, save_roc
from scripts.train import get_test_predictions

batch_size = 0


def save_test_metrics(data_loader, i, net_g, net_d,
                      criterionGAN, criterionFeat, criterionVGG,
                      params_loss, to_neptune, lambda_feat, num_D, n_layers_D):
    for batch in data_loader:
        real_a, real_b, labels = batch[0].cuda(), batch[1].cuda(),\
                                check_label(batch[2].cuda())
        _ = make_forward((real_a, real_b, labels), net_g, net_d,
                         (criterionGAN, criterionFeat, criterionVGG),
                         params_loss, to_neptune,
                         lambda_feat, num_D, n_layers_D)
    neptune.log_metric('val_psnr', i, get_generator_metric(data_loader, net_g))


def make_forward(data, net_g, net_d, criterions, params_loss, to_neptune,
                 lambda_feat, num_D, n_layers_D):
    real_a, real_b, labels = data
    criterionGAN, criterionFeat, criterionVGG = criterions
    loss_return = dict()

    real_a_labels = torch.cat((real_a, labels), 1)
    fake_b = net_g(real_a_labels)

    # forward G with fake
    pred_fake = net_d.forward(torch.cat((real_a, labels, fake_b), 1))
    loss_G_GAN = criterionGAN(pred_fake, True)

    # forward D with fake, detach G
    fake_ab = torch.cat((real_a, labels, fake_b.detach()), 1)
    pred_fake = net_d.forward(fake_ab.detach())
    loss_d_fake = criterionGAN(pred_fake, False)

    # forward D with real
    real_ab = torch.cat((real_a, labels, real_b.detach()), 1)
    pred_real = net_d.forward(real_ab)
    loss_d_real = criterionGAN(pred_real, True)

    # GAN feature matching loss
    loss_G_Feat = 0
    if not params_loss['no_ganFeat_loss']:
        feat_weights = 4.0 / (n_layers_D + 1)
        D_weights = 1.0 / num_D
        for i in range(num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_Feat += D_weights * feat_weights * \
                    criterionFeat(pred_fake[i][j],
                                  pred_real[i][j].detach()) * lambda_feat
        loss_return['loss_g_feat'] = loss_G_Feat

    # VGG feature matching loss
    loss_G_VGG = 0
    if not params_loss['no_vgg_loss']:
        loss_G_VGG = criterionVGG(fake_b, real_b) * lambda_feat
        loss_return['loss_g_vgg'] = loss_G_VGG

    loss_D = (loss_d_fake + loss_d_real) * 0.5
    loss_G = loss_G_GAN + loss_G_Feat + loss_G_VGG

    loss_return.update(dict(loss_g=loss_G, loss_g_gan=loss_G_GAN,
                            loss_d=loss_D, loss_d_fake=loss_d_fake,
                            loss_d_real=loss_d_real))

    if to_neptune:
        neptune.log_metric('g', loss_G.data.cpu().numpy())
        neptune.log_metric('g_gan', loss_G_GAN.data.cpu().numpy())
        neptune.log_metric('g_gan_feat', loss_G_Feat)
        neptune.log_metric('g_vgg', loss_G_VGG.data.cpu().numpy())
        neptune.log_metric('d', loss_D.data.cpu().numpy())
        neptune.log_metric('d0', loss_d_fake.data.cpu().numpy())
        neptune.log_metric('d1', loss_d_real.data.cpu().numpy())

    return loss_return


def train(datas, params_loss, n_epochs=200, lr=.0002, beta1=.5, lambda_feat=10,
          type_G='global', n_downsample_global=3, n_blocks_global=9,
          n_layers_D=3, num_D=3, getIntermFeat=False,
          device=None, verbose=2, name='', to_neptune=True):

    training_data_loader, testing_data_loader, testing_data_loader_bs1 = datas
    add_nc = training_data_loader.dataset[0][2].shape[0]
    print(f'# of channel of additional input vector: {add_nc}')
    add_input = True if add_nc > 0 else False
    net_g = define_G_hd(3+add_nc, 3, type_G='global',
                        norm='batch', device=device)
    net_d = define_D_hd(6+add_nc, n_layers_D=n_layers_D, num_D=3,
                        getIntermFeat=False, norm='batch', device=device)

    if to_neptune:
        neptune.create_experiment(name=name, params={'batch_size': batch_size,
                                                     'lr': lr,
                                                     'lambda_l1': lambda_feat,
                                                     'n_epochs': n_epochs,
                                                     # 'd_patches': patches
                                                     },
                                  upload_source_files=['train.py',
                                                       'params_hd.json',
                                                       'scripts/'])

    criterionGAN = GANLoss_hd()
    criterionFeat = torch.nn.L1Loss()
    if not params_loss['no_vgg_loss']:
        criterionVGG = VGGLoss(device)

    # # setup optimizer
    optimizer_G = optim.Adam(net_g.parameters(), lr=lr,
                             betas=(beta1, 0.999))
    optimizer_D = optim.Adam(net_d.parameters(), lr=lr,
                             betas=(beta1, 0.999))

    for epoch in tqdm(range(1, n_epochs+1)):
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b, labels = batch[0].cuda(), batch[1].cuda(),\
                check_label(batch[2].cuda())

            # ##################### Forward Pass ####################
            losses = make_forward((real_a, real_b, labels), net_g, net_d,
                                  (criterionGAN, criterionFeat, criterionVGG),
                                  params_loss, to_neptune,
                                  lambda_feat, num_D, n_layers_D)
            loss_G = losses['loss_g']
            loss_D = losses['loss_d']

            # ##################### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        if to_neptune:
            i = (epoch-1)*len(training_data_loader)
            neptune.log_metric('psnr', i,
                               get_generator_metric(training_data_loader,
                                                    net_g))
            save_test_metrics(testing_data_loader, i, net_g, net_d,
                              criterionGAN, criterionFeat, criterionVGG,
                              params_loss, to_neptune,
                              lambda_feat, num_D, n_layers_D)

        if epoch % verbose == 0 and to_neptune:
            create_checkpoint(epoch, net_g, net_d)
            plot_pred(training_data_loader, testing_data_loader,
                      net_g, device, epoch)
            # save_roc(training_data_loader, epoch, net_g, net_d,
            #          len(training_data_loader))
            # save_roc(testing_data_loader, epoch, net_g, net_d,
            #          len(training_data_loader), test=True)
    if to_neptune:
        all_pred = get_test_predictions(epoch, testing_data_loader_bs1,
                                        train_set, train_close, add_input)
        neptune.log_image('pred_all', all_pred)
        neptune.stop()


if __name__ == '__main__':
    neptune.init('declot/alpix2pix')

    opt = get_options_from_json(hd=True)
    df = pd.read_csv('/home/refenement/Projects/Dataset_flats/flats_info_ini.csv')
    df_ini = pd.read_csv('/home/refenement/Projects/Dataset_flats/flats_info_ini.csv')

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
    device = [torch.device("cuda:0")]

    train((training_data_loader, testing_data_loader, testing_data_loader_bs1),
          opt['params_loss'], n_epochs=opt['n_epochs'],
          lambda_feat=opt['lambda_feat'],
          type_G=opt['type_G'],
          verbose=opt['verbose'], name=opt['name'], device=device,
          to_neptune=opt['to_neptune'])

    # # get_test_predictions(100, testing_data_loader_bs1, train_set,
    # train_close, device)
   
