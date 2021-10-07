import torch.nn as nn
import torch
from torchsummary import summary
from nets.layer import make_conv1d_layers
# from layer import make_conv1d_layers
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

from torch.nn import functional as F
import os

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo. It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DeConv(nn.Sequential):
    def __init__(self, in_ch, mid_ch, out_ch, norm_layer=None, activation_layer=None):
        super(DeConv, self).__init__(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            norm_layer(mid_ch),
            activation_layer(mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            activation_layer(out_ch),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

class DeConv_v2(nn.Sequential):
    def __init__(self, in_ch, mid_ch, out_ch, norm_layer=None, activation_layer=None):
        super(DeConv_v2, self).__init__(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            norm_layer(mid_ch),
            activation_layer(mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            activation_layer(out_ch),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, activation_layer=None):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(out_planes)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, activation_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, activation_layer=activation_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class LpNetSkiConcat(nn.Module):
    def __init__(self,
                 input_size=(256,256),
                 vertex_num=6890,
                 input_channel = 48,
                 embedding_size = 2048,
                 width_mult=1.0,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 activation_layer=None,
                 inverted_residual_setting=None):

        super(LpNetSkiConcat, self).__init__()

        assert input_size[1] in [256]

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.PReLU # PReLU does not have inplace True
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 64, 1, 2],  #[-1, 48, 64, 64]
                [6, 48, 2, 1],  #[-1, 48, 64, 64]
                [6, 48, 3, 2],  #[-1, 48, 32, 32]
                [6, 64, 4, 1],  #[-1, 64, 32, 32]
                [6, 96, 3, 2],  #[-1, 96, 16, 16]
                [6, 160, 3, 1], #[-1, 160, 16, 16]
                [6, 320, 1, 2], #[-1, 320, 8, 8]
            ]

        self.vertex_num = vertex_num
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        self.first_conv = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=activation_layer)

        inv_residual = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                inv_residual.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, activation_layer=activation_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.inv_residual = nn.Sequential(*inv_residual)

        self.last_conv = ConvBNReLU(input_channel, embedding_size, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer)

        self.deconv0 = DeConv(embedding_size, _make_divisible(inverted_residual_setting[-3][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv1 = DeConv(256, _make_divisible(inverted_residual_setting[-4][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv2 = DeConv(256, _make_divisible(inverted_residual_setting[-5][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        # self.deconv3 = DeConv_v2(256, 64, 256, norm_layer=norm_layer, activation_layer=activation_layer)

        # self.final_layer = nn.Conv2d(
        #     in_channels=256,
        #     out_channels= joint_num * 32,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        self.conv_x = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*64], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size, device=heatmap1d.device).float()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, x):
        x = self.first_conv(x)
        x = self.inv_residual[0:6](x)
        x2 = x
        x = self.inv_residual[6:10](x)
        x1 = x
        x = self.inv_residual[10:13](x)
        x0 = x
        x = self.inv_residual[13:16](x)
        x = self.inv_residual[16:](x) # [1, 320, 4, 4]
        img_feat = self.last_conv(x) # [1, 2048, 8, 8]


        z = torch.cat([x0, img_feat], dim=1)
        z = self.deconv0(z)
        z = torch.cat([x1, z], dim=1)
        z = self.deconv1(z)
        z = torch.cat([x2, z], dim=1) #[1, 304, 16, 16]
        img_feat_xy = self.deconv2(z) # [1, 256, 32, 32]
        # z = self.final_layer(z)
        # img_feat_xy = self.deconv3(z) # [1, 256, 64, 64]

        # x axis
        img_feat_x = img_feat_xy.mean((2)) #[1, 256, 64]
        heatmap_x = self.conv_x(img_feat_x) # [1, 6890, 64]
        coord_x = self.soft_argmax_1d(heatmap_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None] # [1, 2048, 1]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,64)
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        mesh_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return mesh_coord

    # def forward(self, x):
    #     img_feat = self.forward_feature(x)


    def init_weights(self):
        for i in [self.deconv0, self.deconv1, self.deconv2]:
            for name, m in i.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for j in [self.first_conv, self.inv_residual, self.last_conv, self.final_layer]:
            for m in j.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if hasattr(m, 'bias'):
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # LpNetSkiConcat((256, 256), 18).init_weights()
    model = LpNetSkiConcat()
    model = model.cuda()
    model.eval()
    test_data = torch.rand(1, 3, 256, 256).cuda()
    test_outputs = model(test_data)
    # print(test_outputs.size())
    # summary(model, (3, 256, 256))
    flops = FlopCountAnalysis(model, test_data)
    print(flop_count_table(flops))