import sys
sys.path.append('../')

import torch
import torch.nn as nn
from networks.init_weight import *
from networks.utils_layers import ASPP, MLP, conv_block, up_conv, AxialPositionalEmbedding, AxialAttention, DSV
import math
import torch.nn.functional as F

class SingleEmbedding(nn.Module):

    def __init__(self, in_ch, img_size, patch_size, embed_dim=384) -> None:
        super(SingleEmbedding, self).__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])

        R = 256 // img_size
        r = 256 // (16 * R)

        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=(1,1), padding=0)

        self.avgPool = nn.AvgPool2d(kernel_size=r, stride=r)

    def forward(self, x):

        x = self.proj(x)
        x = self.avgPool(x)        
        # x = x.flatten(2).transpose(1,2)
        return x


class AxialTransformerBlock(nn.Module):

    def __init__(self, embed_dim, attention_heads) -> None:
        super().__init__()

        self.axial_attention = AxialAttention(
            dim=embed_dim,
            num_dimensions=3,
            heads=attention_heads,
            dim_heads=None,
            dim_index=1,
            sum_axial_out=True
        )

        self.MLP = MLP(embedding_dim=embed_dim, mlp_dim=embed_dim * 4)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        '''
        param: x
        shape: (batch, 384, z, 16, 16)
        return: (batch, 384, z, 16, 16)
        '''
        b, embed_dim, slices, p_h, p_w = x.size()

        _x = self.axial_attention(x)
        _x = self.dropout(_x)
        x = x + _x

        x = x.flatten(2).transpose(1, 2) # (batch, 384, 3, 16, 16) -> (batch, 768, 384)
        x = self.layer_norm1(x)

        _x = self.MLP(x)
        x = x + _x
        x = self.layer_norm2(x)

        f_reshape = x.permute(0, 2, 1).contiguous().view(b, embed_dim, slices, p_h, p_w)

        return f_reshape

class AxialTransformer(nn.Module):
    def __init__(self, embed_dim, attention_heads, depth=4):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [AxialTransformerBlock(embed_dim, attention_heads) for _ in range(depth)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x
        
class AATM(nn.Module):

    def __init__(self, in_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums=None) -> None:
        super(AATM, self).__init__()

        self.single_slice_embedding = SingleEmbedding(in_ch, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.pos_embed = AxialPositionalEmbedding(
            dim=embed_dim, 
            shape=(sample_slices, patch_size, patch_size), 
            emb_dim_index=1
        )

        index = int(math.log2(256 // img_size))
        depth = block_nums[index]

        self.axial_transformer = AxialTransformer(embed_dim, attention_heads, depth)
        
    def forward(self, features):

        '''
        features: (batch, channel, z, x, y)

        return: (batch, channel, z, x, y)
        '''

        slices_num = features.shape[2]

        embedding_features = [self.single_slice_embedding(features[:,:,i]).unsqueeze(2) for i in range(slices_num)]
        
        f_qkv = torch.cat(embedding_features, dim=2) # (batch, 384, 3, 16, 16)
        f_embed = self.pos_embed(f_qkv)
        
        attention_output = self.axial_transformer(f_embed)

        return attention_output
    

class AxialBlock(nn.Module):

    def __init__(self, in_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums):
        super(AxialBlock, self).__init__()

        self.img_size = (img_size, img_size)

        self.sample_slices = sample_slices

        self.AATM = AATM(in_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums)

        self.conv3D = nn.Conv3d(embed_dim, in_ch, kernel_size=(sample_slices, 1, 1), stride=1, padding=0)

        self.conv_1x1 = nn.Conv3d(embed_dim, 1, kernel_size=(1, 1, 1), stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()
                
        self.slice_weight = nn.Parameter(torch.ones([sample_slices//2, 256, 256])/2, requires_grad=False)
        
        if sample_slices >= 5 :
            for i in range(sample_slices//2 - 1):
                self.slice_weight[i] = 0.0

    def forward(self, features):
        
        '''
        features: type: []; size: 3, which are lower, key, upper slices, seperately.
        '''

        f_expand = [f.unsqueeze(2) for f in features]

        f_cat = torch.cat(f_expand, dim=2)

        f_AATM = self.AATM(f_cat)


        f_fuse = self.conv3D(f_AATM).squeeze()
        # 
        if f_AATM.shape[0] == 1:
            f_fuse = f_fuse.unsqueeze(0)    
        f_reshape = F.interpolate(f_fuse, self.img_size, mode='bilinear')

        '''
        generate the pseudo label
        '''

        f_auxiliary = self.conv_1x1(f_AATM).squeeze()
        if f_AATM.shape[0] == 1:
            f_auxiliary = f_auxiliary.unsqueeze(0)  
        assert len(f_auxiliary.shape) == 4

        f_auxiliary = F.interpolate(f_auxiliary, (256, 256),  mode='bilinear') # 修改：每个分支都插值到256 256

        f_prop = self.sigmoid(f_auxiliary)

        f_prop = self.get_pseudo_label(f_prop)

        return f_reshape, f_prop

    def get_pseudo_label(self, f_prop):

        false_gt = torch.zeros(*f_prop[:,0,...].shape).cuda()

        for i in range(self.sample_slices//2):
            false_gt += self.slice_weight[i] * (f_prop[:,i,...]+f_prop[:,self.sample_slices-i-1,...])

        false_gt[false_gt>1.0] = 1.0
        mid_slice = f_prop[:,self.sample_slices//2,...]

        assert false_gt.shape == mid_slice.shape

        return [mid_slice, false_gt]


class BackBone(nn.Module):

    def __init__(self, in_ch, out_ch, img_size, sample_slices=3, patch_size=16, embed_dim=384, attention_heads=12, block_nums=[2,2,4,2]) -> None:
        super(BackBone, self).__init__()

        self.conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                
            )] * sample_slices)
        
        self.axial_block = AxialBlock(out_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums)

    def forward(self, x):
        '''
        param: x : [] * 3; []->shape: (batch, channel, h, w)
        '''
        features = [slice_conv(x[idx]) for idx, slice_conv in enumerate(self.conv)]
        
        f_AATM, f_prop = self.axial_block(features)

        return features, f_AATM, f_prop
    

class Encoder(nn.Module):
    
    def __init__(self, in_ch, out_ch) -> None:
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x, axial_feature):
        x = self.conv1(x)
        x = x + axial_feature
        out = self.conv2(x)
        return out


class Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Bottleneck, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(out_ch, out_ch, atrous_rates=[6, 12, 18])
        
    def forward(self, x):

        out = self.conv(x)
        out = self.aspp(out)
        return out
    

class TAGNet(nn.Module):

    def __init__(self, img_ch, out_ch=1, img_size=256, dsv=False, sample_slices=3) -> None:
        super(TAGNet, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        self.dsv = dsv

        filters = [64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters] # [16, 32, 64, 128, 256]

        self.backbone1 = BackBone(in_ch=img_ch, out_ch=filters[0], img_size=img_size, sample_slices=sample_slices) # 1->16->16
        self.encoder1 = Encoder(in_ch=img_ch*sample_slices, out_ch=filters[0]) # 3->16->16

        self.backbone2 = BackBone(in_ch=filters[0], out_ch=filters[1], img_size=img_size//2, sample_slices=sample_slices) # 16->32->32
        self.encoder2 = Encoder(in_ch=filters[0], out_ch=filters[1]) # 16->32->32

        self.backbone3 = BackBone(in_ch=filters[1], out_ch=filters[2], img_size=img_size//4, sample_slices=sample_slices)
        self.encoder3 = Encoder(in_ch=filters[1], out_ch=filters[2])

        self.backbone4 = BackBone(in_ch=filters[2], out_ch=filters[3], img_size=img_size//8, sample_slices=sample_slices)
        self.encoder4 = Encoder(in_ch=filters[2], out_ch=filters[3])

        self.encoder5 = Bottleneck(in_ch=filters[3], out_ch=filters[4])

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_conv5 = up_conv(in_ch=filters[4], out_ch=filters[3])
        self.up_conv4 = up_conv(in_ch=filters[3], out_ch=filters[2])
        self.up_conv3 = up_conv(in_ch=filters[2], out_ch=filters[1])
        self.up_conv2 = up_conv(in_ch=filters[1], out_ch=filters[0])

        self.decoder4 = conv_block(in_ch=filters[4], out_ch=filters[3])
        self.decoder3 = conv_block(in_ch=filters[3], out_ch=filters[2])
        self.decoder2 = conv_block(in_ch=filters[2], out_ch=filters[1])
        self.decoder1 = conv_block(in_ch=filters[1], out_ch=filters[0])

        self.dsv4 = DSV(in_channel=filters[3], out_channel=out_ch, scale_factor=8)
        self.dsv3 = DSV(in_channel=filters[2], out_channel=out_ch, scale_factor=4)
        self.dsv2 = DSV(in_channel=filters[1], out_channel=out_ch, scale_factor=2)
        self.dsv1 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        '''最后的1X1卷积'''
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weight)


    def forward(self, x):
        
        slices_num = None
        
        if len(x.shape) == 4:
            slices_num = x.shape[1]
            _x = x.unsqueeze(1)
            input = [_x[:,:,i,...] for i in range(slices_num)]
        
        # encoder
        features1, f_aatm1, slice_prop_1 = self.backbone1(input)
        res_encoder1 = self.encoder1(x, f_aatm1)
        features1_pool = [self.Maxpool(features1[i]) for i in range(slices_num)]
        x2 = self.Maxpool(res_encoder1)
        
        features2, f_aatm2, slice_prop_2 = self.backbone2(features1_pool)
        res_encoder2 = self.encoder2(x2, f_aatm2)
        features2_pool = [self.Maxpool(features2[i]) for i in range(slices_num)]
        x3 = self.Maxpool(res_encoder2)

        features3, f_aatm3, slice_prop_3 = self.backbone3(features2_pool)
        res_encoder3 = self.encoder3(x3, f_aatm3)
        features3_pool = [self.Maxpool(features3[i]) for i in range(slices_num)]
        x4 = self.Maxpool(res_encoder3) 

        features4, f_aatm4, slice_prop_4 = self.backbone4(features3_pool)
        res_encoder4 = self.encoder4(x4, f_aatm4)
        # features4_pool = [self.Maxpool(features4[i]) for i in range(slices_num)]
        x5 = self.Maxpool(res_encoder4) 

        res_encoder5 = self.encoder5(x5) # 最后一次无池化 torch.Size([8, 1024, 16, 16])
        
        # deocer and contact

        de4 = self.up_conv5(res_encoder5)
        de4 = torch.cat((res_encoder4, de4), dim=1)
        de4 = self.decoder4(de4)

        de3 = self.up_conv4(de4)
        de3 = torch.cat((res_encoder3, de3), dim=1)
        de3 = self.decoder3(de3)

        de2 = self.up_conv3(de3)
        de2 = torch.cat((res_encoder2, de2), dim=1)
        de2 = self.decoder2(de2)

        de1 = self.up_conv2(de2)
        de1 = torch.cat((res_encoder1, de1), dim=1)
        de1 = self.decoder1(de1)

        if self.dsv is True:
            dsv4 = self.dsv4(de4)
            output4 = self.sigmoid(dsv4)

            dsv3 = self.dsv3(de3)
            output3 = self.sigmoid(dsv3)

            dsv2 = self.dsv2(de2)
            output2 = self.sigmoid(dsv2)

            dsv1 = self.dsv1(de1)
            output1 = self.sigmoid(dsv1)
            return [output4, output3, output2, output1], [slice_prop_1, slice_prop_2, slice_prop_3, slice_prop_4]
            
        else:
            final = self.final(de1)
            output = self.sigmoid(final)
            return [output, [slice_prop_1, slice_prop_2, slice_prop_3, slice_prop_4]]



def get_model(img_size, dsv=False, sample_slices=3):

    print('use deep supervision: {}'.format(dsv))
    model = TAGNet(1, 1, img_size=img_size, dsv=dsv, sample_slices=sample_slices)
    return model


if __name__ == '__main__':

    x = torch.rand([8, 5, 256, 256])

    model = get_model(256, False, 5)

    output, slice_prop = model(x)

    print(output.shape)
    for prop in slice_prop:
        print(prop[0].shape, prop[1].shape)

    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))