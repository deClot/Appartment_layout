import torch
import torch.nn as nn
from torch.nn import init
import functools

# ########------------------ GENERATOR ------------------############


def encoder_block(in_c, out_c, f=4, s=2, p=1,
                  reflection_pad=False, batchnorm=True,
                  activation='leaky_relu', name=None):
    block = nn.Sequential()
    if reflection_pad:
        block.add_module('%s_refl_pad' % name, nn.ReflectionPad2d(3))

    block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c,
                                                 kernel_size=f, stride=s,
                                                 padding=p, bias=False))
    if batchnorm:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if activation == 'relu':
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    return block


def decoder_block(in_c, out_c, f=4, s=2, p=2,
                  batchnorm=True, dropout=True, activation='relu', name=None):
    block = nn.Sequential()
    block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c,
                                                           kernel_size=f,
                                                           stride=s,
                                                           padding=p,
                                                           bias=False))
    block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if activation == 'relu':
        block.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    elif activation == 'tanh':
        block.add_module('%s_tanh' % name, nn.Tanh())
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))

    return block


def define_Ublock(in_c, out_c, name, transposed=False,
                  batchnorm=True, relu=True, tanh=False, dropout=False):
    block = nn.Sequential()
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c,
                                                     kernel_size=4, stride=2,
                                                     padding=1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c,
                                                               kernel_size=4, stride=2,
                                                               padding=1, bias=False))
    if batchnorm:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    elif tanh:
        block.add_module('%s_tanh' % name, nn.Tanh())
    else:
        block.add_module('%s_leakyrelu'% name, nn.LeakyReLU(0.2, inplace=True))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class U_Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()

        self.encoder1 = encoder_block(input_nc, 64, batchnorm=False,
                                      name='enc1')            # 256-128
        self.encoder2 = encoder_block(64, 128, name='enc2')   # 128-64
        self.encoder3 = encoder_block(128, 256, name='enc3')  # 64-32
        self.encoder4 = encoder_block(256, 512, name='enc4')  # 32-16
        self.encoder5 = encoder_block(512, 512, name='enc5')  # 16-8
        self.encoder6 = encoder_block(512, 512, name='enc6')  # 8-4
        self.encoder7 = encoder_block(512, 512, name='enc7')  # 4-2

        self.bottleneck = encoder_block(512, 512, batchnorm=False, name='neck')  # 2-1

        self.decoder7 = decoder_block(512, 512, name='dec7')    # 1-2
        self.decoder6 = decoder_block(512*2, 512, name='dec6')  # 2-4
        self.decoder5 = decoder_block(512*2, 512, name='dec5')  # 4-8
        self.decoder4 = decoder_block(512*2, 512, dropout=False, name='dec4')  # 8-16
        self.decoder3 = decoder_block(512*2, 256, dropout=False, name='dec3')  # 16-32
        self.decoder2 = decoder_block(256*2, 128, dropout=False, name='dec2')  # 32-64
        self.decoder1 = decoder_block(128*2, 64, dropout=False, name='dec1')   # 64-128

        self.out = decoder_block(64*2, 3, batchnorm=False, dropout=False,
                                 activation='tanh', name='out')
        print(self.encoder1)

    def forward(self, x):
        out1 = self.encoder1(x)     # 256-128
        out2 = self.encoder2(out1)  # 128-64
        out3 = self.encoder3(out2)  # 64-32
        out4 = self.encoder4(out3)  # 32-16
        out5 = self.encoder5(out4)  # 16-8
        out6 = self.encoder6(out5)  # 8-4
        out7 = self.encoder7(out6)  # 4-2

        out_neck = self.bottleneck(out7)   # 2-1

        dout7 = self.decoder7(out_neck)    # 1-2
        dout7_out7 = torch.cat([dout7, out7], 1)  # 2*2
        dout6 = self.decoder6(dout7_out7)  # 2-4
        dout6_out6 = torch.cat([dout6, out6], 1)  # 4*4
        dout5 = self.decoder5(dout6_out6)  # 4-8
        dout5_out5 = torch.cat([dout5, out5], 1)  # 8*8
        dout4 = self.decoder4(dout5_out5)  # 8-16
        dout4_out4 = torch.cat([dout4, out4], 1)  # 16*16
        dout3 = self.decoder3(dout4_out4)  # 16-32
        dout3_out3 = torch.cat([dout3, out3], 1)  # 32*32
        dout2 = self.decoder2(dout3_out3)  # 32-64
        dout2_out2 = torch.cat([dout2, out2], 1)  # 64*64
        dout1 = self.decoder1(dout2_out2)  # 64-128
        dout1_out1 = torch.cat([dout1, out1], 1)  # 128*128

        out = self.out(dout1_out1)         # 128-256
        return out


class Res_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,  n_blocks=6,
                 norm_layer=nn.BatchNorm2d, use_dropout=True, use_bias=False):
        assert(n_blocks >= 0)
        super().__init__()
        # 256-262-256, 3-64
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # add downsampling layers
        # devide img shape by 2 (256-128-64)
        # increase number of filters in power of 2 (64-128-256)
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        padding_type = 'reflect'
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout,
                                  use_bias=False)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, nc, padding_type, norm_layer, activation=nn.ReLU(True),
                 use_dropout=True, use_bias=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(nc, padding_type, norm_layer,
                                                use_dropout, use_bias)

    def build_conv_block(self, nc, padding_type, norm_layer,
                         use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            nc (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented'
                                      % padding_type)

        conv_block += [nn.Conv2d(nc, nc, kernel_size=3, padding=p, bias=False),
                       norm_layer(nc), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented'
                                      % padding_type)

        conv_block += [nn.Conv2d(nc, nc, kernel_size=3, padding=p, bias=False),
                       norm_layer(nc)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


# ########------------------ DISCRIMINATOR ------------------############

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # first layer without batchnorm
        # 256*256*(3+3) -> 128*128*64
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]

        # n_layers usuall blocks Conv-BN-LRelu
        # n_leyrs=3: 128*128*64 ->1. 64*64*128 ->2. 32*32*256 ->3. 16*16*512
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)  # max number of chnnels = 512
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # additiona conv with stride=1, doesn't change output size
        # 16*16*512 -> 15*15*512
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # convert to 14*14*1
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



