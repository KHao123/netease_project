import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
import torchgeometry as tgm
from nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers
# from layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers
import os

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        self.deconv = make_deconv_layers([2048,256,256,256])
        self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return joint_coord

class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64+joint_num*cfg.output_hm_shape[0],64])

    def forward(self, img_feat, joint_heatmap_3d):
        joint_heatmap_3d = joint_heatmap_3d.view(-1,self.joint_num*cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        feat = torch.cat((img_feat, joint_heatmap_3d),1)
        feat = self.conv(feat)
        return feat

class MeshNet(nn.Module):
    def __init__(self, vertex_num):
        super(MeshNet, self).__init__()
        self.vertex_num = vertex_num
        self.deconv = make_deconv_layers([2048,256,256,256])
        self.conv_x = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        mesh_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return mesh_coord

class ParamRegressor(nn.Module):
    def __init__(self, joint_num):
        super(ParamRegressor, self).__init__()
        self.joint_num = joint_num
        self.fc = make_linear_layers([self.joint_num*3, 1024, 512], use_bn=True)
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.fc_pose = make_linear_layers([512, 16*6], relu_final=False) # hand joint orientation
        else:
            self.fc_pose = make_linear_layers([512, 24*6], relu_final=False) # body joint orientation
        self.fc_shape = make_linear_layers([512, 10], relu_final=False) # shape parameter

    def rot6d_to_rotmat(self,x):
        x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pose_3d):
        batch_size = pose_3d.shape[0]
        pose_3d = pose_3d.view(-1,self.joint_num*3)
        feat = self.fc(pose_3d)

        pose = self.fc_pose(feat)
        pose = self.rot6d_to_rotmat(pose)
        pose = torch.cat([pose,torch.zeros((pose.shape[0],3,1)).cuda().float()],2)
        pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(batch_size,-1)
        
        shape = self.fc_shape(feat)

        return pose, shape

class DeConv(nn.Sequential):
    def __init__(self, in_ch, mid_ch, out_ch, norm_layer=None, activation_layer=None):
        super(DeConv, self).__init__(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            norm_layer(mid_ch),
            activation_layer(mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            activation_layer(out_ch),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

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

class PoseNet_small(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet_small, self).__init__()
        self.joint_num = joint_num
        inverted_residual_setting = [
                # t, c, n, s
                [1, 64, 1, 1],  #[-1, 48, 256, 256]
                [6, 48, 2, 2],  #[-1, 48, 128, 128]
                [6, 48, 3, 2],  #[-1, 48, 64, 64]
                [6, 64, 4, 2],  #[-1, 64, 32, 32]
                [6, 96, 3, 2],  #[-1, 96, 16, 16]
                [6, 160, 3, 2], #[-1, 160, 8, 8]
                [6, 320, 1, 1], #[-1, 320, 8, 8]
            ]
        width_mult = 1.0
        round_nearest=8
        norm_layer = nn.BatchNorm2d
        activation_layer = nn.PReLU
        # self.deconv = make_deconv_layers([512,256])
        self.deconv0 = DeConv(2048, _make_divisible(inverted_residual_setting[-2][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv1 = DeConv(256, _make_divisible(inverted_residual_setting[-3][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv2 = DeConv(256, _make_divisible(inverted_residual_setting[-4][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)



        self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        # img_feat_xy = self.deconv(img_feat) # [32, 512, 32, 32] -> [32, 256, 64, 64]
        img_feat_xy = self.deconv0(img_feat)
        img_feat_xy = self.deconv1(img_feat_xy)
        img_feat_xy = self.deconv2(img_feat_xy)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x) # [32, 29, 64]
        coord_x = self.soft_argmax_1d(heatmap_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return joint_coord

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

class Pose2Feat_small(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat_small, self).__init__()
        self.joint_num = joint_num
        # self.conv = make_conv_layers([24+joint_num*cfg.output_hm_shape[0],24])
        inplane = 24+joint_num*cfg.output_hm_shape[0]
        self.conv = nn.Sequential(
            ConvBNReLU(inplane, inplane, stride=1, groups=inplane, norm_layer=nn.BatchNorm2d, activation_layer=nn.PReLU),
            nn.Conv2d(inplane, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24)
        )

    def forward(self, img_feat, joint_heatmap_3d): 
        #[1, 24, 64, 64]+[1, 29, 64, 64, 64] ->   [1, 32, 128, 128]
        joint_heatmap_3d = joint_heatmap_3d.view(-1,self.joint_num*cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        feat = torch.cat((img_feat, joint_heatmap_3d),1)
        feat = self.conv(feat)
        return feat

class MeshNet_small(nn.Module):
    def __init__(self, vertex_num):
        super(MeshNet_small, self).__init__()
        self.vertex_num = vertex_num
        # self.deconv = make_deconv_layers([512,256])
        inverted_residual_setting = [
                # t, c, n, s
                [1, 64, 1, 1],  #[-1, 48, 256, 256]
                [6, 48, 2, 2],  #[-1, 48, 128, 128]
                [6, 48, 3, 2],  #[-1, 48, 64, 64]
                [6, 64, 4, 2],  #[-1, 64, 32, 32]
                [6, 96, 3, 2],  #[-1, 96, 16, 16]
                [6, 160, 3, 2], #[-1, 160, 8, 8]
                [6, 320, 1, 1], #[-1, 320, 8, 8]
            ]
        width_mult = 1.0
        round_nearest=8
        norm_layer = nn.BatchNorm2d
        activation_layer = nn.PReLU
        # self.deconv = make_deconv_layers([512,256])
        self.deconv0 = DeConv(2048, _make_divisible(inverted_residual_setting[-2][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv1 = DeConv(256, _make_divisible(inverted_residual_setting[-3][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv2 = DeConv(256, _make_divisible(inverted_residual_setting[-4][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)



        self.conv_x = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        # img_feat_xy = self.deconv(img_feat)
        img_feat_xy = self.deconv0(img_feat)
        img_feat_xy = self.deconv1(img_feat_xy)
        img_feat_xy = self.deconv2(img_feat_xy)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        mesh_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return mesh_coord


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= '6'
    # model =  PoseNet_small(29).cuda()
    # x = torch.rand([4, 512, 32, 32]).cuda()
    # x = model(x) #[4, 29, 3]
    # print(x.shape)

    model = Pose2Feat_small(29)
    x = torch.rand([4, 48, 128, 128])
    y = torch.rand([4, 29, 64, 64, 64])
    x = model(x, y)
    print(x.shape) #[4, 48, 128, 128]

    # model =  MeshNet_small(6890).cuda()
    # x = torch.rand([4, 512, 32, 32]).cuda()
    # x = model(x) #[4, 6890, 3]
    # print(x.shape)