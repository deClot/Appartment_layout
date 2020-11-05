import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models

import neptune
from sklearn.metrics import roc_auc_score

from scripts.datasets import postproces_img, get_sample_by_name


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer, epoch, show=True):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    if show:
        neptune.log_metric('lr', lr)
#         print(f'epoch {epoch} >> learning rate = %.7f' % lr)


def create_checkpoint(epoch, net_g, net_d):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    net_g_model_out_path = "checkpoint/netG_model_epoch_{}.pth".format(epoch)
    net_d_model_out_path = "checkpoint/netD_model_epoch_{}.pth".format(epoch)
    torch.save(net_g, net_g_model_out_path)
    torch.save(net_d, net_d_model_out_path)
    # print(f"epoch {epoch} >> Checkpoint saved to {'checkpoint'}")


def check_label(label):
    if label.ndim == 2:
        return label.squeeze()
    else:
        return label


def get_generator_metric(data_loader, net_g):
    avg_psnr = 0
    criterionMSE = nn.MSELoss()
    for batch in data_loader:
        input, target, labels = batch[0].cuda(), batch[1].cuda(),\
                                            check_label(batch[2].cuda())
        input = torch.cat((input, labels), 1)
        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * np.log10(1 / mse.item())
        avg_psnr += psnr
    return avg_psnr / len(data_loader)


def get_gan_metric(data_loader, net_g, net_d):
    def get_d_pred(a, b, label):
        ab = torch.cat((a, b, label), 1)
        pred = net_d(ab)
        return pred,# pred.sum()/pred.numel()

    # metrics_fake, metrics_real = [], []
    pred_fake_all, pred_real_all = [], []
    names = []
    for batch in data_loader:
        real_a, real_b, label, name = batch[0].cuda(),\
                                      batch[1].cuda(),\
                                      check_label(batch[2].cuda()),\
                                      batch[3]
        names.append(name)
        fake_b = net_g(torch.cat((real_a, label), 1))
        pred_fake, metric_fake = get_d_pred(real_a, fake_b, label)
        pred_real, metric_real = get_d_pred(real_a, real_b, label)
        assert (pred_fake == pred_real).sum().item() != pred_fake.numel()
        # metrics_fake.append(metric_fake.view(-1).tolist())
        # metrics_real.append(metric_real.view(-1).tolist())
        pred_fake_all.extend(pred_fake.view(-1).tolist())
        pred_real_all.extend(pred_real.view(-1).tolist())

    return metric_real, metric_fake,  pred_real_all, pred_fake_all, names


def save_roc(data_loader, epoch, net_g, net_d,
             len_train_dataset, test=False):
    _, _, pred_real, pred_fake, names = get_gan_metric(data_loader,
                                                       net_g, net_d)
    true = np.concatenate((np.ones((len(pred_real), 1)),
                           np.zeros((len(pred_fake), 1))), axis=0)
    pred = np.concatenate((pred_real, pred_fake), axis=0)
    auc_total = roc_auc_score(true, pred)
    name_metric = 'val_' if test else ''
    neptune.log_metric(name_metric + 'auc_total',
                       (epoch-1)*len_train_dataset, auc_total)

    size = int(len(pred_real)/len(data_loader))
    assert true.shape[0] == pred.shape[0]
    for i in range(len(data_loader)):
        sample = np.concatenate([pred_real[i*size:(i+1)*size],
                                pred_fake[i*size:(i+1)*size]])
        true = np.concatenate([np.ones((size,)),
                               np.zeros((size,))])
        tmp = f'{epoch}\t{names[i]}:\n{roc_auc_score(true, sample)}'
        neptune.log_text(name_metric+'auc_sample', tmp)


def get_data_to_plot(data_loader, net_g):
    real_a, real_b, label, name = next(data_loader)
    real_a = real_a.cuda()
    label = check_label(label.cuda())
    fake_b = net_g(torch.cat((real_a, label), 1))

    nrooms = [1, 1]
    if label.ndim == 4:
        if label[0, 1, 0, 0] != 1:  # change appartment to 2 rooms
            nrooms[1] = 2
            if label[0, 2, 0, 0] == 1:
                nrooms[0] = 3
            else:
                nrooms[0] = 1
            label[:, 1, :, :] = 1
            label[:, 2, :, :] = 0
        else:
            nrooms = [2, 1]
            label[:, 1, :, :] = 0  # all 2rooms will be 1room

    fake_b2 = net_g(torch.cat((real_a, label), 1))
    return (real_a, real_b, fake_b, fake_b2), nrooms, name[0]


def plot_sample(axes, data, name_type, nrooms, name, epoch):
    real_a, real_b, fake_b, fake_b2 = data
    if real_a.ndim == 4:
        real_a = real_a[0]
        real_b = real_b[0]
        fake_b = fake_b[0]
        fake_b2 = fake_b2[0]

    t = postproces_img(real_a)
    axes[0].axis('off')
    axes[0].imshow(t)
    axes[0].set_title(name_type)

    t = postproces_img(real_b)
    axes[1].axis('off')
    axes[1].imshow(t)
    axes[1].set_title(name + f'_epoch{epoch}')

    t = postproces_img(fake_b)
    axes[2].axis('off')
    axes[2].imshow(t)
    axes[2].set_title(f'nrooms = {nrooms[0]}')

    # 1 - n rooms 2, 2 - n rooms 3

    t = postproces_img(fake_b2)
    axes[3].axis('off')
    axes[3].imshow(t)
    axes[3].set_title(f'nrooms = {nrooms[1]}')
    return axes


def plot_pred(training_data_loader, testing_data_loader,
              net_g, device, epoch):
    plt.ioff()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    data, nrooms, name = get_data_to_plot(iter(training_data_loader), net_g)
    axes[0] = plot_sample(axes[0], data, 'train', nrooms, name, epoch)
    data, nrooms, name = get_data_to_plot(iter(testing_data_loader), net_g)
    axes[1] = plot_sample(axes[1], data, 'test', nrooms, name, epoch)

    neptune.log_image('pred', fig)
    del fig, axes


def get_test_predictions(epoch, data_loader, train_set, train_close,
                         add_input=True):
    net_g_model_out_path = "checkpoint/netG_model_epoch_{}.pth".format(epoch)
#     net_d_model_out_path = "checkpoint/netD_model_epoch_{}.pth".format(epoch)
    g = torch.load(net_g_model_out_path)
    print(f'Loaded model from : {net_g_model_out_path}')
#     torch.save(net_d, net_d_model_out_path)
    data_loader = iter(data_loader)
    n_rows = len(data_loader)
    fig, axes = plt.subplots(nrows=n_rows, ncols=6, figsize=(15, 17))
    for i in range(n_rows):
        data, nrooms, name = get_data_to_plot(data_loader, g)
        axes[i][:4] = plot_sample(axes[i][:4], data, 'test', nrooms, name, epoch)
        train_1, train_21, label, _ = \
            get_sample_by_name(train_set, train_close[i])
        axes[i][4].imshow(postproces_img(train_21))
        axes[i][4].axis('off')
        axes[i][4].set_title('train')

        label = label.unsqueeze(0) if add_input else label
        train_pred = g(torch.cat((train_1.unsqueeze(0).cuda(),
                                  label.cuda()), 1))
        axes[i][5].imshow(postproces_img(train_pred[0]))
        axes[i][5].axis('off')
    return fig


def get_scheduler(optimizer, lr_policy, epochs_info):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    epoch_count, n_epochs, n_epochs_decay, lr_decay_iters = epochs_info
    if lr_policy == 'linear':
        def lambda_rule(epoch):  # current + starting - total number
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla',
                 target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCELoss()  # nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GANLoss_hd(nn.Module):
    def __init__(self, use_lsgan=True,
                 target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss_hd, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i]*self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
