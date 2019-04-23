# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.softmax = nn.Softmax(dim=0)
        self.classifier_1 = nn.Linear(2048, self.num_classes)
        self.bnc1 = nn.BatchNorm1d(2048)
        self.classifier_2 = nn.Linear(512, self.num_classes)
        self.bnc2 = nn.BatchNorm1d(512)
        self.classifier_3 = nn.Linear(128, self.num_classes)
        self.bnc3 = nn.BatchNorm1d(128)
        self.reccur = nn.Linear(self.in_planes, self.in_planes)
        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


    def _generate_SA(self, output):
        output = torch.sum(output, dim=1)
        output_size = output.size()
        # print("start for size")
        # print(output_size)
        output_flatten = self.softmax(output.view(output_size[0], -1))
        # print(output_flatten.size())
        output_2d = output_flatten.view(output_size[0], output_size[1], output_size[2])
        # print("output from _generate_SA")
        # print(output_2d.size())
        output = output_2d*output
        output = output.view(output.size()[0], -1)
        # print("final size")
        # print(output.size())
        return output


    def forward(self, x):

        # x = self.base(x)[-1]
        output_2, output_3, x = self.base(x)
        # output_4 = self._generate_SA(output_4)
        # print("output size that i want to see")
        # print(x.size())
        # global_feat = self.classifier_inter(x)
        # global_feat = self.gap(x)  # (b, 2048, 1, 1)
        # print(global_feat.size())
        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        for i in range(4):
          global_feat = self.reccur(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            # print("BNNECK PRESENT")
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            # output_1 = self.classifier_1(self.bnc1(self._generate_SA(output_1)))
            output_2 = self.classifier_2(self.bnc2(self._generate_SA(output_2)))
            output_3 = self.classifier_3(self.bnc3(self._generate_SA(output_3)))
            cls_score = self.classifier(feat)
            return output_2, output_3, cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # if 'classifier' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])