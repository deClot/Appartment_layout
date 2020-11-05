import functools

import torch
import torch.nn as nn
from torch.nn import init

from scripts.models import Res_Generator, U_Generator, NLayerDiscriminator
from scripts.models_hd import GlobalGenerator, LocalEnhancer, Encoder
from scripts.models_hd import MultiscaleDiscriminator


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                       track_running_stats=False)
    else:
        raise NotImplementedError(f'normalization layer {norm_type} is not found')
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func> trought all layers


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids='cuda:0'):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
#     if len(gpu_ids) > 0:
    assert(torch.cuda.is_available())
    net.to(gpu_ids)
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, type_G, norm='batch', use_dropout=True,
             init_type='normal', init_gain=0.02, device='cuda:0'):
    print(norm)
    norm_layer = get_norm_layer(norm_type=norm)
    if type_G == 'resnet_9blocks':
        net = Res_Generator(input_nc, output_nc, norm_layer=norm_layer,
                            use_dropout=use_dropout, n_blocks=9)
    elif type_G == 'resnet_6blocks':
        net = Res_Generator(input_nc, output_nc, norm_layer=norm_layer,
                            use_dropout=use_dropout, n_blocks=6)
    elif type_G == 'unet_256':
        net = U_Generator(input_nc, output_nc, 8, norm_layer=norm_layer,
                          use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized'
                                  % type_G)
    return init_net(net, init_type, init_gain, device)


def define_G_hd(input_nc, output_nc, ngf=64, type_G='global',
                n_downsample_global=3, n_blocks_global=9,
                n_local_enhancers=1, n_blocks_local=3,
                norm='instance', init_type='normal', init_gain=0.02,
                device=['cuda:0']):
    norm_layer = get_norm_layer(norm_type=norm)
    if type_G == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf,
                               n_downsample_global, n_blocks_global, norm_layer)
    elif type_G == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf,
                             n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    elif type_G == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf,
                       n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    # print(netG)
    if len(device) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(device[0])
    init_weights(netG, init_type, init_gain=init_gain)
    return netG


def define_D(input_nc, ndf=64, netD='basic', n_layers=3, use_sigmoid=True,
             norm='batch', init_type='normal', init_gain=0.02, gpu_id=None):
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3,
                                  norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers,
                                  norm_layer=norm_layer, use_sigmoid=use_sigmoid)
#     elif netD == 'pixel':
#         net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_id)


def define_D_hd(input_nc, ndf=64, n_layers_D=3, num_D=3, getIntermFeat=False,
                norm='instance', use_sigmoid=True,
                init_type='normal', init_gain=0.02, device=['cuda:0']):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer,
                                   use_sigmoid, num_D, getIntermFeat)
    # print(netD)
    if len(device) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(device[0])
    init_weights(netD, init_type, init_gain=init_gain)
    return netD
