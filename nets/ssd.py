import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from nets.mobilenetv2 import InvertedResidual, mobilenet_v2
from nets.vgg import vgg as add_vgg
from nets.resnet import resnet50
#from deform_conv_v2 import DeformConv2d
from nets.dilated_upsample import dilated_conv_upsample, FusionLayer, FusionLayer1
from nets.CAattention import CA_Block
from nets.DCNV2_conv_pool import DeformConv2d,DeformMaxPool2d, DeformAvgPool2d

# VGG架构...
'''
0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                            ->300, 300, 64
1 ReLU(inplace=True)
2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                           ->300, 300, 64
3 ReLU(inplace=True)
4 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                  ->150, 150, 64
5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                          ->150, 150, 128
6 ReLU(inplace=True)
7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                         ->150, 150, 128
8 ReLU(inplace=True)
9 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                  ->75, 75, 128
10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->75, 75, 256
11 ReLU(inplace=True)
12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->75, 75, 256
13 ReLU(inplace=True)
14 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->75, 75, 256
15 ReLU(inplace=True)
16 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)                  ->38, 38, 256
17 Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->38, 38, 512 [计划选中]
18 ReLU(inplace=True)
19 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->38, 38, 512 [计划选中]
20 ReLU(inplace=True)
21 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->38, 38, 512 [选中]
22 ReLU(inplace=True)
23 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                 ->19, 19, 512
24 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->19, 19, 512
25 ReLU(inplace=True)
26 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->19, 19, 512
27 ReLU(inplace=True)
28 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        ->19, 19, 512
29 ReLU(inplace=True)
30 MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)                 ->19, 19, 512
31 Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))      ->19, 19, 1024 [计划选中]
32 ReLU(inplace=True)
33 Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))                                      ->19, 19, 1024 [选中]
34 ReLU(inplace=True)

38, 38, 512的序号是22
19, 19, 1024的序号是34
'''
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def add_extras(in_channels, backbone_name):
    layers = []
    if backbone_name == 'mobilenetv2':
        layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
        layers += [CA_Block(channel=512, reduction=16, out_channels=512)]

        layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
        layers += [CA_Block(channel=256, reduction=16, out_channels=256)]

        layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
        layers += [CA_Block(channel=256, reduction=16, out_channels=256)]

        layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]
        layers += [CA_Block(channel=64, reduction=16, out_channels=64)]
    else:
        # Block 6
        # 19,19,1024 -> 19,19,256 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        # layers += [CA_Block(channel=256, reduction=16, out_channels=256)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
        #layers += [CA_Block(channel=512, reduction=16, out_channels=512)]

        # Block 7
        # 10,10,512 -> 10,10,128 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        # layers += [CA_Block(channel=128, reduction=16, out_channels=128)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        #layers += [CA_Block(channel=256, reduction=16, out_channels=256)]

        # Block 8
        # 5,5,256 -> 5,5,128 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        # layers += [CA_Block(channel=128, reduction=16, out_channels=128)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        #layers += [CA_Block(channel=256, reduction=16, out_channels=256)]

        # Block 9
        # 3,3,256 -> 3,3,128 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        # layers += [CA_Block(channel=128, reduction=16, out_channels=128)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        #layers += [CA_Block(channel=256, reduction=16, out_channels=256)]

    return nn.ModuleList(layers)

class SSD300(nn.Module):

    def __init__(self, num_classes, backbone_name, pretrained=False):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        # 注意力  可变形卷积================================
        self.conv1_51264 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv1_102464 = nn.Conv2d(1024, 64, kernel_size=1)
        self.conv1_25664 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv1_64256 = nn.Conv2d(64, 256, kernel_size=1)
        self.conv1_64512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv1_641024 = nn.Conv2d(64, 1024, kernel_size=1)
        self.attention1 = CA_Block(512)#resnet50 vgg 512 #mobilenetv2 96
        self.attention2 = CA_Block(1024)#resnet50 vgg 1024 #mobilenetv2 1280
        self.conv_171921 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv_3133 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        # self.deform_conv = DeformConv2d(512, 512, kernel_size=3, padding=1, modulation=True)
        # self.deform_conv2 = DeformConv2d(1024, 1024, kernel_size=3, padding=1, modulation=True)
        self.deform_conv = DeformConv2d(256, 256, kernel_size=3, padding=1, modulation=True)
        self.deform_max_pool1 = DeformMaxPool2d(64, 64, kernel_size=38, padding=0, stride=1,modulation=True)#512, 256
        #self.deform_max_pool1 = DeformMaxPool2d(512, 256, kernel_size=38, padding=0, stride=1, modulation=True)
        self.deform_avg_pool1 = DeformAvgPool2d(64, 64, kernel_size=38, padding=0, stride=1,modulation=True)
        #self.deform_avg_pool1 = DeformAvgPool2d(512, 256, kernel_size=38, padding=0, stride=1, modulation=True)

        self.deform_max_pool2 = DeformMaxPool2d(64, 64, kernel_size=7, padding=1, stride=7, modulation=True)#1024, 256
        #self.deform_max_pool2 = DeformMaxPool2d(1024, 256, kernel_size=7, padding=0, stride=6, modulation=True)
        self.deform_avg_pool2 = DeformMaxPool2d(64, 64, kernel_size=7, padding=1, stride=7, modulation=True)
        #self.deform_avg_pool2 = DeformAvgPool2d(1024, 256, kernel_size=7, padding=0, stride=6, modulation=True)

        self.deform_max_pool3 = DeformMaxPool2d(64, 64, kernel_size=2, padding=0, stride=2, modulation=True)#512, 256
        #self.deform_max_pool3 = DeformMaxPool2d(512, 256, kernel_size=3, padding=0, stride=2, modulation=True)
        self.deform_avg_pool3 = DeformAvgPool2d(64, 64, kernel_size=2, padding=0, stride=2, modulation=True)
        #self.deform_avg_pool3 = DeformAvgPool2d(512, 256, kernel_size=3, padding=0, stride=2, modulation=True)

        # =================================
        if backbone_name == "vgg":
            self.vgg = add_vgg(pretrained)
            self.extras = add_extras(1024, backbone_name)
            self.L2Norm = L2Norm(512, 20)
            mbox = [4, 6, 6, 6, 4, 4]

            loc_layers = []
            conf_layers = []
            backbone_source = [21, -2]
            # ---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            # ---------------------------------------------------#
            for k, v in enumerate(backbone_source):

                loc_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            # -------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#
            for k, v in enumerate(self.extras[1::2], 2):  #未使用注意力时，self.extras[1::2], 2===========有 self.extras[2::3], 2(额外层的注意力放在中间) ===有self.extras[2::3], 2(注意力放最后)

                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == "mobilenetv2":
            self.mobilenet = mobilenet_v2(pretrained).features
            self.extras = add_extras(1280, backbone_name)#1280
            self.L2Norm = L2Norm(96, 20)#(96, 20）
            mbox = [6, 6, 6, 6, 6, 6]#[6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [
                    nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras[1::2], 2):   #无注意力(self.extras, 2)====有(self.extras[1::2], 2)
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == "resnet50":
            self.resnet = nn.Sequential(*resnet50(pretrained).features)
            self.extras = add_extras(1024, backbone_name)
            self.L2Norm = L2Norm(512, 20)
            mbox = [4, 6, 6, 6, 4, 4]

            loc_layers = []
            conf_layers = []
            out_channels = [512, 1024]
            # ---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第layer3层和layer4层可以用来进行回归预测和分类预测。
            # ---------------------------------------------------#
            for k, v in enumerate(out_channels):
                loc_layers += [nn.Conv2d(out_channels[k], mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(out_channels[k], mbox[k] * num_classes, kernel_size=3, padding=1)]
            # -------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        else:
            raise ValueError("The backbone_name is not support")

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        self.backbone_name = backbone_name

    def forward(self, x):
        # ---------------------------#
        #   x是300,300,3
        # ---------------------------#
        #===========================
        # 计划将融合前特征层添加到shallow list中，然后通过浅层特征增强，输出增强后的的特征图到sources中，之后不变
        shallow1 = list()
        shallow2 = list()
        sources0 = list()
        # 定义一个空列表，用于存储应用了 CA_Block 注意力模块后的特征层
        sources_with_ca = []
        # ===========================
        sources = list()
        loc = list()
        conf = list()
        # ---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        # ---------------------------#
        if self.backbone_name == "vgg":
            for k in range(19):     #修改前23  后19
                x = self.vgg[k](x)
                #==================
            shallow1.append(x)
            for k in range(19, 21):
                x = self.vgg[k](x)
            shallow1.append(x)
            for k in range(21, 23):
                x = self.vgg[k](x)
            shallow1.append(x)
                #====================
        elif self.backbone_name == "mobilenetv2":
            for k in range(14):
                x = self.mobilenet[k](x)
        elif self.backbone_name == "resnet50":
            for k in range(6):
                x = self.resnet[k](x)
        # 特征增强==========
        # 第3层 + 每层通过3x3 这4个进行逐元素相加赋值给x
        feature_map_0 = shallow1[2]
        for i in range(3):
            shallow1[i] = self.conv_171921(shallow1[i])
        feature_map_1 = shallow1[0]
        feature_map_2 = shallow1[1]
        feature_map_3 = shallow1[2]
        merged_feature_map1 = feature_map_0 + feature_map_1 + feature_map_2 + feature_map_3
        x = merged_feature_map1
        #x = torch.from_numpy(x)

        #x = self.attention1(x)
        # ==============

        # ---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化
        # ---------------------------#
        s = self.L2Norm(x)
        #y = self.L2Norm(x) #s = self.L2Norm(x)
        #=====可变形卷积
        #s = self.deform_conv1(y)
        #======
        sources0.append(s)#sources.append(s)

        # ---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        # ---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23, 33): #原range(23, len(self.vgg)): 改23，33
                x = self.vgg[k](x)
            #===========
            shallow2.append(x)
            for k in range(33, len(self.vgg)):
                x = self.vgg[k](x)
            shallow2.append(x)
            #===========
        elif self.backbone_name == "mobilenetv2":
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == "resnet50":
            for k in range(6, len(self.resnet)):
                x = self.resnet[k](x)
        # 特征增强==========
        # 第2层 + 每层通过3x3 这3个进行逐元素相加赋值给x
        feature_map_4 = shallow2[1]
        for i in range(2):
            shallow2[i] = self.conv_3133(shallow2[i])
        feature_map_5 = shallow2[0]
        feature_map_6 = shallow2[1]
        merged_feature_map2 = feature_map_4 + feature_map_5 + feature_map_6
        x = merged_feature_map2

        #x = torch.from_numpy(x)

        # x = self.attention2(x)
        # s = self.deform_conv1(x)
        # ==============
        sources0.append(x)#sources.append(s)
        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg" or self.backbone_name == "resnet50":
                if k % 2 == 1 :     #无k % 2 == 1     有 k % 3 == 2(额外层的注意力放在中间)      有 k % 3 == 2(额外层的注意力放在最后)
                    sources0.append(x)#sources.append(x)
            else:
                # sources.append(x)

                if k % 2 == 1:
                    sources.append(x)#   有用if k % 2 == 1:   无 不用

        # # 打印 sources0 中各个特征层的形状
        # for i, feature_layer in enumerate(sources0):
        #     print(f"Shape of feature layer {i + 1}: {feature_layer.shape}")

        #====sources0进行融合放入sources中====可变形和CA加最后
        #   1,3,4   2,4,5   3,5,6
        # 第一层操作[1]
        layer1 = sources0[0]
        layer1_64 = self.conv1_51264(layer1)
        layer1_dilated_64 = dilated_conv_upsample(layer1_64, dilation_rate=5, upsample_size=(38, 38))
        layer1_dilated = self.conv1_64512(layer1_dilated_64)
        # 第二层操作[2]
        layer2 = sources0[1]
        layer2_64 = self.conv1_102464(layer2)
        layer2_dilated_64 = dilated_conv_upsample(layer2_64, dilation_rate=5, upsample_size=(19, 19))
        layer2_dilated = self.conv1_641024(layer2_dilated_64)
        # 第三层操作[3]
        layer3 = sources0[2]
        layer3_64 = self.conv1_51264(layer3)
        layer3_dilated_64 = dilated_conv_upsample(layer3_64, dilation_rate=3, upsample_size=(38, 38))
        layer3_dilated = self.conv1_641024(layer3_dilated_64)
        # 第四层操作[4]
        layer4 = sources0[3]
        layer4_dilated = dilated_conv_upsample(layer4, dilation_rate=1, upsample_size=(38, 38))

        # 融合第一组特征层[1,3,4]
        fused1 = FusionLayer(
            layer1.shape[1] + layer1_dilated.shape[1] + layer3_dilated.shape[1] + layer4_dilated.shape[1])(
            torch.cat([layer1, layer1_dilated, layer3_dilated, layer4_dilated], dim=1))
        fusedCA1 = self.attention1(fused1)
        sources.insert(0, fusedCA1)
        #print(f"fusedCA1 Shape: {fusedCA1.shape}")
        # 第四层操作，二次操作[4]
        layer42 = sources0[3]
        layer42_dilated = dilated_conv_upsample(layer42, dilation_rate=3, upsample_size=(19, 19))

        # 第五层操作[5]
        layer5 = sources0[4]
        layer5_dilated = dilated_conv_upsample(layer5, dilation_rate=1, upsample_size=(19, 19))

        # 融合第二组特征层[2,4,5]
        fused2 = FusionLayer1(
            layer2.shape[1] + layer2_dilated.shape[1] + layer42_dilated.shape[1] + layer5_dilated.shape[1])(
            torch.cat([layer2, layer2_dilated, layer42_dilated, layer5_dilated], dim=1))
        fusedCA2 = self.attention2(fused2)
        sources.append(fusedCA2)
        #print(f"fusedCA2 Shape: {fusedCA2.shape}")
        # 第三层操作，二次操作[3]
        layer32 = sources0[2]
        layer32_dilated = dilated_conv_upsample(layer32, dilation_rate=5, upsample_size=(10, 10))

        # 第五层操作，，二次操作[5]
        layer52 = sources0[4]
        layer52_dilated = dilated_conv_upsample(layer52, dilation_rate=3, upsample_size=(10, 10))

        # 第六层操作[6]
        layer6 = sources0[5]
        layer6_dilated = dilated_conv_upsample(layer6, dilation_rate=1, upsample_size=(10, 10))

        # 融合第三组特征层[3,5,6]
        fused3 = FusionLayer(
            layer3.shape[1] + layer32_dilated.shape[1] + layer52_dilated.shape[1] + layer6_dilated.shape[1])(
            torch.cat([layer3, layer32_dilated, layer52_dilated, layer6_dilated], dim=1))
        fusedCA3 = self.attention1(fused3)
        sources.append(fusedCA3)
        #print(f"fusedCA3 Shape: {fusedCA3.shape}")
        # # 循环遍历 sources 中的每个特征层
        # for feature_layer in sources:
        #     # 创建 CA_Block 注意力模块并应用于当前特征层
        #     ca_block = CA_Block(channel=feature_layer.shape[1], reduction=16)
        #     feature_layer_with_ca = ca_block(feature_layer)
        #
        #     # 将应用了 CA_Block 注意力模块的特征层添加到新的列表
        #     sources_with_ca.append(feature_layer_with_ca)
        #
        # # 将新的列表 sources_with_ca 替换原始列表 sources
        # sources = sources_with_ca

        # #=上述完成了前三层+注意力====下面完成后三层+可变形

        sources.append(layer4)
        sources.append(layer5)
        sources.append(layer6)
        # # [4]
        # layer3_conv1 = self.conv1_51264(layer3)
        # layer3_conv1_max3 = self.deform_max_pool3(layer3_conv1)
        # layer3_conv1_avg3 = self.deform_avg_pool3(layer3_conv1)
        # layer3_max3 = self.conv1_64256(layer3_conv1_max3)
        # layer3_avg3 = self.conv1_64256(layer3_conv1_avg3)
        #
        # fused_layer4 = layer3_max3 + layer3_avg3 + layer4
        # deform_used_layer4 = self.deform_conv(fused_layer4)
        # # print(f"deform_used_layer4 Shape: {deform_used_layer4.shape}")
        # sources.append(deform_used_layer4)
        #
        # # [5]
        # layer2_conv1 = self.conv1_102464(layer2)
        # layer2_conv1_max2 = self.deform_max_pool2(layer2_conv1)
        # layer2_conv1_avg2 = self.deform_avg_pool2(layer2_conv1)
        # layer2_max2 = self.conv1_64256(layer2_conv1_max2)
        # layer2_avg2 = self.conv1_64256(layer2_conv1_avg2)
        #
        # fused_layer5 = layer2_max2 + layer2_avg2 + layer5
        # deform_used_layer5 = self.deform_conv(fused_layer5)
        # # print(f"deform_used_layer5 Shape: {deform_used_layer5.shape}")
        # sources.append(deform_used_layer5)
        #
        #
        # #[6]
        # layer1_conv1 = self.conv1_51264(layer1)
        # layer1_conv1_max1 = self.deform_max_pool1(layer1_conv1)
        # layer1_conv1_avg1 = self.deform_avg_pool1(layer1_conv1)
        # layer1_max1 = self.conv1_64256(layer1_conv1_max1)
        # layer1_avg1 = self.conv1_64256(layer1_conv1_avg1)
        #
        # fused_layer6 = layer1_max1 + layer1_avg1 + layer6
        # deform_used_layer6 = self.deform_conv(fused_layer6)
        # # print(f"deform_used_layer6 Shape: {deform_used_layer6.shape}")
        # sources.append(deform_used_layer6)





        # for i, feature_layer in enumerate(sources):
        #     print(f"sources Shape of feature layer {i + 1}: {feature_layer.shape}")

        #=============



        # -------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        # -------------------------------------------------------------#
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # -------------------------------------------------------------#
        #   进行reshape方便堆叠
        # -------------------------------------------------------------#
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # -------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        # -------------------------------------------------------------#
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output
