import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        # g = torch.tanh(cc_g)
        g = torch.sigmoid(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * F.relu(c_next)
        # changed from tanh to relu
        return h_next, c_next

    def init_hidden(self, batch_size, dev):
        # height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=dev),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=dev))

class GC(nn.Module):
    """https://github.com/seoungwugoh/RGMP/blob/master/model.py"""
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class EncoderExtra(nn.Module):
    def __init__(self, encoder, out_ch=2048, pretrained=True, freeze_bn=False):
        super(EncoderExtra, self).__init__()
        encoder = encoder(pretrained=pretrained)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = encoder.conv1
        self.bn = encoder.bn1
        self.relu = encoder.relu

        self.mask_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # extra layer that goes with the pool
        # conv+bn+relu+maxpool
        self.added_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.added_bn = nn.BatchNorm2d(64)

        self.conv2 = encoder.layer1
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4
        self.reduce_channel_ = nn.Conv2d(2048, out_ch, 1)
        self.reduce_bn = nn.BatchNorm2d(out_ch)

        for m in [self.mask_conv, self.added_conv, self.reduce_channel_]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, prev_mask):
        conv1 = self.relu(self.bn(self.conv1(x) + self.mask_conv(prev_mask))) # 64x112 *2
        added_layer = F.relu(self.added_bn(self.added_conv(conv1)))
        next_res = self.pool(added_layer)
        conv2 = self.conv2(next_res) # 64x112
        conv3 = self.conv3(conv2) # 32x56
        conv4 = self.conv4(conv3) # 16x28
        conv5 = self.conv5(conv4) # 8x14
        out = F.relu(self.reduce_bn(self.reduce_channel_(conv5)))

        return out, conv5, conv4, conv3, conv2, added_layer,


class DecoderRef(nn.Module):
    def __init__(self,
                num_ch,
                num_classes,
                out_ch=1):
        super(DecoderRef, self).__init__()
        self.GC_p = GC(inplanes=2*num_ch, planes=num_ch//2, kh=7, kw=7)
        self.GCGC_p = nn.Conv2d(num_ch, num_ch//2, 1)

        self.GC = GC(inplanes=2*num_ch, planes=num_ch//2, kh=7, kw=7)

        self.conv_1 = nn.Conv2d(num_ch//2+2048, 1024, 1)
        self.conv_11 = nn.Conv2d(1024, 512, 5, padding=2)

        self.GCmm0 = GC(inplanes=2*num_ch, planes=num_ch//2, kh=7, kw=7)
        self.GCmm_1 = GC(inplanes=2*num_ch, planes=num_ch//2, kh=7, kw=7)
        self.mm = nn.Conv2d(num_ch, num_ch//2, 1)

        self.conv_2_ = nn.Conv2d(512+1024+512, 512, 1)
        self.conv_22 = nn.Conv2d(512, 256, 5, padding=2)

        self.conv_3 = nn.Conv2d(256+512, 256, 1)
        self.conv_33 = nn.Conv2d(256, 128, 5, padding=2)

        self.conv_4 = nn.Conv2d(128+256, 128, 1)
        self.conv_44 = nn.Conv2d(128, 64, 5, padding=2)

        self.conv_5 = nn.Conv2d(64+64, 64, 1)
        self.conv_55 = nn.Conv2d(64, 64, 5, padding=2)

        self.distance_classifier_ = nn.Conv2d(64, num_classes, 3, padding=1)
        self.segmentation_branch_ = nn.Conv2d(64, num_classes, 3, padding=1)
        self.merge_ = nn.Conv2d(2 * num_classes, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, ref, h, mid_features, previous_f, h_sm, mid_ref):

        x = F.relu(self.GC(torch.cat([ref, h], dim=1)))
        y = F.relu(self.GC_p(torch.cat([previous_f, h], dim=1)))
        x = F.relu(self.GCGC_p(torch.cat([x, y], dim=1)))

        x = torch.cat((x, mid_features[0]), dim=1)
        x = F.relu(self.conv_1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv_11(x))

        mm0 = F.relu(self.GCmm0(torch.cat([mid_ref[0], h_sm], dim=1)))
        mm_1 = F.relu(self.GCmm_1(torch.cat([mid_ref[1], h_sm], dim=1)))
        mm = F.relu(self.mm(torch.cat([mm0, mm_1], dim=1)))
        x = torch.cat((x, mid_features[1], mm), dim=1)

        x = F.relu(self.conv_2_(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv_22(x))

        x = torch.cat((x, mid_features[2]), dim=1)
        x = F.relu(self.conv_3(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv_33(x))

        x = torch.cat((x, mid_features[3]), dim=1)
        x = F.relu(self.conv_4(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv_44(x))

        x = torch.cat((x, mid_features[4]), dim=1)
        x = F.relu(self.conv_5(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = F.relu(self.conv_55(x))
        class_scores = self.distance_classifier_(x)
        seg_branch = self.segmentation_branch_(x)
        pred_mask = self.merge_(F.relu(torch.cat([class_scores, seg_branch], dim=1)))
        return pred_mask, class_scores 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ModelMatch(nn.Module):
    def __init__(self,
            epsilon,
            backbone=models.resnet101,
            input_size=(8, 14),
            num_ch=1024,
            mem_kernel=(3, 3),
            input_size_sm=(16, 28),
            sm_kernel=(5,5),
            num_classes=None):

        super(ModelMatch, self).__init__()
        self.epsilon = epsilon

        self.encoder_ref = EncoderExtra(encoder=backbone, out_ch=num_ch)
        self.encoder = EncoderExtra(encoder=backbone, out_ch=num_ch)

        self.memory = ConvLSTMCell(input_size=input_size, input_dim=num_ch,
                                 hidden_dim=num_ch, kernel_size=mem_kernel, bias=True)

        self.skip_memory = ConvLSTMCell(input_size=input_size_sm, input_dim=num_ch,
                                 hidden_dim=num_ch, kernel_size=sm_kernel, bias=True)

        self.decoder = DecoderRef(num_ch, num_classes)

    def forward(self, rgb, gt, epoch=None, offset=None, mode='train'):
        predicted_masks = []
        h, c = self.memory.init_hidden(rgb[0].size(0), device)
        h_sm, c_sm = self.skip_memory.init_hidden(rgb[0].size(0), device)

        if mode=='train':
            ref_feats, _, m0, _, _, _ = self.encoder_ref(rgb[0].to(device), gt[0].to(device))
            epsilon = max(self.epsilon, 1. - 0.004*(epoch-offset))

            for ii in range(len(rgb) - 1):
                if ii > 0 and np.random.uniform() > epsilon and epoch > offset:
                    temp = torch.sigmoid(decoded_imgs[0])
                    prev_f,_ ,m1 ,_ , _, _ = self.encoder_ref(rgb[ii].to(device), temp)
                    enc = self.encoder(rgb[ii+1].to(device), temp)

                else:
                    prev_f,_ ,m1 ,_ , _, _ = self.encoder_ref(rgb[ii].to(device), gt[ii].to(device))
                    enc = self.encoder(rgb[ii+1].to(device), gt[ii].to(device))

                h, c = self.memory(enc[0], (h, c))
                h_sm, c_sm = self.skip_memory(enc[2], (h_sm, c_sm))
                decoded_imgs = self.decoder(ref_feats, h, enc[1:], prev_f, h_sm, [m0, m1])
                predicted_masks.append(decoded_imgs)
        else:
            ref_feats, _, m0, _, _, _ = self.encoder_ref(rgb[0].to(device), gt.to(device))
            for ii in range(len(rgb) - 1):
                if ii==0:
                    prev_f,_ ,m1 ,_ , _, _ = self.encoder_ref(rgb[ii].to(device), gt.to(device))
                    enc = self.encoder(rgb[ii+1].to(device), gt.to(device))
                else:
                    temp = torch.sigmoid(decoded_imgs[0])
                    assert temp.shape==gt.shape
                    prev_f,_ ,m1 ,_ , _, _ = self.encoder_ref(rgb[ii].to(device), temp)
                    enc = self.encoder(rgb[ii+1].to(device), temp)

                h, c = self.memory(enc[0], (h, c))
                h_sm, c_sm = self.skip_memory(enc[2], (h_sm, c_sm))
                decoded_imgs = self.decoder(ref_feats, h, enc[1:], prev_f, h_sm, [m0, m1])
                predicted_masks.append(decoded_imgs[0])

        return predicted_masks


EPS = 1e-15
class SegmentationLossWithJaccardIndexLoss(nn.BCEWithLogitsLoss):

    def __init__(self, pos_weight, jacc_weight=0.3):
        super(SegmentationLossWithJaccardIndexLoss, self).__init__()
        self.jacc_weight = jacc_weight
        self.pos_weight = pos_weight
    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target.float(), pos_weight=self.pos_weight)

        jaccard_target = (target == 1).float()
        jaccard_output = torch.sigmoid(output)
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        jacc = torch.log((intersection + EPS) / (union - intersection + EPS))
        return (1 - self.jacc_weight) * bce - jacc * self.jacc_weight

class JaccardIndexLoss(nn.BCEWithLogitsLoss):

    def __init__(self):
        super(JaccardIndexLoss, self).__init__()
    def forward(self, output, target):

        jaccard_target = (target == 1).float()
        jaccard_output = torch.sigmoid(output)
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        jacc = torch.log((intersection + EPS) / (union - intersection + EPS))
        return -jacc


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """https://github.com/kmaninis/OSVOS-PyTorch/blob/master/layers/osvos_layers.py"""

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss

