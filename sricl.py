import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from skimage import measure
import numpy as np


def generate_contour(mask, kernel_size=3, iterations=1):
    assert mask.dim() == 4 and mask.shape[1] == 1, "Mask must be of shape B1HW"
    
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32).to(mask.device)
    
    eroded = mask.clone()
    for _ in range(iterations):
        eroded = F.conv2d(eroded, kernel, padding=kernel_size//2)
        eroded = (eroded == kernel_size**2).float()
    
    dilated = mask.clone()
    for _ in range(iterations):
        dilated = F.conv2d(dilated, kernel, padding=kernel_size//2)
        dilated = (dilated > 0).float()
    
    contour = dilated - eroded
    
    return contour

def find_bbox(mask):
    mask = mask.float()
    nonzero_corrds = torch.where(mask == 1)
    min_x = torch.min(nonzero_corrds[1])
    min_y = torch.min(nonzero_corrds[0])
    max_x = torch.max(nonzero_corrds[1])
    max_y = torch.max(nonzero_corrds[0])
    return min_x, min_y, max_x, max_y


def draw_rectangle(mask, bbox):
    b, h, w = mask.shape
    batch_images = torch.zeros_like(mask.float())
    for i in range(b):
        batch_box = bbox[i]
        for j in range(len(batch_box)):
            min_x, min_y, max_x, max_y = batch_box[j]
            batch_images[i, min_y:max_y + 1, min_x:max_x + 1] = 1
    return batch_images


def find_connect_area(mask):
    batch_bbox = []
    for i in range(mask.shape[0]):
        label_mask, num_components = measure.label(mask[i].cpu(), connectivity=1, return_num=True)
        bbox = []
        for connected_label in range(1, num_components + 1):
            component_corrds = torch.where(torch.from_numpy(label_mask) == connected_label)
            min_x = torch.min(component_corrds[1])
            min_y = torch.min(component_corrds[0])
            max_x = torch.max(component_corrds[1])
            max_y = torch.max(component_corrds[0])
            bbox.append((min_x, min_y, max_x, max_y))
        batch_bbox.append(bbox)
    return batch_bbox

def mask_to_points(mask, scale_factor=0.005, dilation=3):
    batch_points = torch.zeros_like(mask.float())

    for i in range(mask.shape[0]):
        label_mask, num_components = measure.label(mask[i].cpu().numpy(), connectivity=1, return_num=True)

        for connected_label in range(1, num_components + 1):
            component_coords = torch.where(torch.from_numpy(label_mask) == connected_label)
            area = component_coords[0].size(0)
            num_points = max(int(area * scale_factor), 1)

            selected_indices = np.random.choice(len(component_coords[0]), num_points, replace=False)
            selected_points = [(component_coords[0][idx], component_coords[1][idx]) for idx in selected_indices]

            for y, x in selected_points:
                batch_points[i, y, x] = 1

    kernel = torch.ones((1, 1, dilation, dilation), dtype=torch.float32, device=batch_points.device)
    dilated_points = F.conv2d(batch_points.unsqueeze(1), kernel, padding=dilation//2).squeeze(1)

    dilated_mask = (dilated_points > 0).float()

    return dilated_mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, x_dim, y_dim=None, heads=8, hid_dim=64, dropout=0., use_sdpa=True):
        super().__init__()
        y_dim = y_dim if y_dim else x_dim
        self.heads = heads
        assert hid_dim % heads == 0
        dim_head = hid_dim // heads
        self.scale = dim_head ** -0.5
        self.use_sdpa = use_sdpa
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(x_dim, hid_dim, bias=False)
        self.to_k = nn.Linear(y_dim, hid_dim, bias=False)
        self.to_v = nn.Linear(y_dim, hid_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(hid_dim, x_dim), nn.Dropout(dropout))

    def forward(self, q, kv):
        # q, kv: L,B,C
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)
        q, k, v = map(lambda t: rearrange(t, 'n b (h d) -> b h n d', h=self.heads), (q, k, v))
        
        if self.use_sdpa:
            # q = q * self.scale
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> n b (h d)')
        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, heads, hid_dim, dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, hid_dim=hid_dim, dropout=dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=hid_dim, dropout=dropout)

    def forward(self, tgt, memory):
        tgt = tgt + self.attn(tgt, memory)
        tgt = self.attn_norm(tgt)
        tgt = tgt + self.ffn(tgt)
        tgt = self.ffn_norm(tgt)
        return tgt

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SRICL(nn.Module):
    def __init__(self, model='base'):
        super().__init__()
        # timm.list
        ###############################Transition Layer########################################
        if model == 'base':
            self.bkbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=True)
            channel = 64
            self.channel = channel

            squeeze_channel = 64
            self.upconv5 = nn.ConvTranspose2d(1024, squeeze_channel, kernel_size=2, stride=2)
            self.conv5_1 = nn.Conv2d(512+squeeze_channel, squeeze_channel, kernel_size=3, padding=1)
            self.bn5_1 = nn.BatchNorm2d(squeeze_channel)
            
            self.upconv4 = nn.ConvTranspose2d(squeeze_channel, squeeze_channel, kernel_size=2, stride=2)
            self.conv4_1 = nn.Conv2d(256+squeeze_channel, squeeze_channel, kernel_size=3, padding=1)
            self.bn4_1 = nn.BatchNorm2d(squeeze_channel)
            
            self.upconv3 = nn.ConvTranspose2d(squeeze_channel, squeeze_channel, kernel_size=2, stride=2)
            self.conv3_1 = nn.Conv2d(128+squeeze_channel, 64, kernel_size=3, padding=1)
            self.bn3_1 = nn.BatchNorm2d(64)

            self.emb_dim = channel
            self.output_dim = 1

            total_dim = self.emb_dim * self.output_dim
            self.ref_proj = nn.Sequential(nn.Linear(1024, 584), nn.LayerNorm(584))
            self.head = nn.Sequential(
                            BasicConv2d(channel, channel, kernel_size=3, padding=1),
                            nn.Dropout2d(p=0.1), 
                            nn.Conv2d(channel, 1, 1)
                        )
        self.cross_atten3 = CrossTransformer(self.emb_dim, 4, hid_dim=self.emb_dim, dropout=0.1)

    def forward(self, x, filter_list, mask_list):
        input = x
        B1, _, _, _ = input.size()
        E2, E3, E4, E5 = self.bkbone(x)

        # ------------Reffering information generation----------------------
        kernels_3 = []
        for i in range(len(filter_list)):  # => 3(within task),3,H,W
            with torch.no_grad():
                self.bkbone.eval()
                _, _, _, feat = self.bkbone(filter_list[i])  # B,C,H,W
                self.bkbone.train()
            B, C, H, W = feat.shape
            memory = feat.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1)  # BHW,1,2048
            memory = self.ref_proj(memory)  # BHW,1,C
            query = memory.reshape(B, H, W, -1)
            mask = F.upsample(mask_list[i], size=feat.size()[2:], mode='nearest')
            mask = mask.reshape(B, H, W, 1)
            query_fore = (mask * query).sum((0, 1, 2)) / (1 + mask.sum((0, 1, 2)))  # C
            query_back = ((1 - mask) * query).sum((0, 1, 2)) / (1 + (1 - mask).sum((0, 1, 2)))  # C

            query = torch.stack((query_fore, query_back), dim=0).unsqueeze(1)  # 2,1,C
            query_3 = self.cross_atten3(query, memory)
            query_3 = query_3.reshape(2, 1, 584, 1, 1)

            kernels_3.append(query_3)
        # ------------Reffering information generation----------------------
        D5 = self.upconv5(E5)
        D5 = torch.cat([D5, E4], dim=1)
        D5 = F.relu(self.bn5_1(self.conv5_1(D5)))
        
        D4 = self.upconv4(D5)
        D4 = torch.cat([D4, E3], dim=1)
        D4 = F.relu(self.bn4_1(self.conv4_1(D4)))
        
        D3 = self.upconv3(D4)
        D3 = torch.cat([D3, E2], dim=1)
        D2 = F.relu(self.bn3_1(self.conv3_1(D3)))

        out_fore = self.head(D2)
        out_fore = F.interpolate(out_fore, size=input.size()[2:], mode='bilinear')

        output_fpn = []
        output_bkg = []
        for k3 in kernels_3:
            dk = k3[0] # 1 584 1 1
            dk1 = dk[:, 0:512, :, :].reshape(8, 64, 1, 1) # 1 512 1 1
            dk2 = dk[:, 512:512+64, :, :].reshape(8, 8, 1, 1)
            dk3 = dk[:, 512+64:, :, :].reshape(1, 8, 1, 1)
            out = F.conv2d(input=D2, weight=dk1, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk2, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk3, stride=1, padding=0)
            output_fpn.append(out)
            dk = k3[1] # 1 584 1 1
            dk1 = dk[:, 0:512, :, :].reshape(8, 64, 1, 1) # 1 512 1 1
            dk2 = dk[:, 512:512+64, :, :].reshape(8, 8, 1, 1)
            dk3 = dk[:, 512+64:, :, :].reshape(1, 8, 1, 1)
            out = F.conv2d(input=D2, weight=dk1, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk2, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk3, stride=1, padding=0)
            output_bkg.append(out)
        output_fpn = torch.cat(output_fpn, dim=1)
        output_bkg = torch.cat(output_bkg, dim=1)
        output_prior = F.interpolate(output_fpn, size=input.size()[2:], mode='bilinear')
        output_priorb = F.interpolate(output_bkg, size=input.size()[2:], mode='bilinear')

        B, C, H, W = E5.shape
        memory = E5.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1)  # BHW,1,2048
        memory = self.ref_proj(memory)  # BHW,1,C
        memory = memory.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        cache = F.interpolate(torch.cat((output_bkg, output_fpn), dim=1), size=memory.size()[2:], mode='bilinear')
        # -----------------SR------------------------------------------------------------------------
        SRFP, SRBP = SR(memory, cache) # B C 1 1 -> 1 C 1 1
        output_fpn = []
        output_bkg = []
        for bs in range(D2.shape[0]):
            dk = k3[0] * 0.7 + SRFP[bs].unsqueeze(0) * 0.3 # 1 584 1 1
            dk1 = dk[:, 0:512, :, :].reshape(8, 64, 1, 1) # 1 512 1 1
            dk2 = dk[:, 512:512+64, :, :].reshape(8, 8, 1, 1)
            dk3 = dk[:, 512+64:, :, :].reshape(1, 8, 1, 1)
            out = F.conv2d(input=D2[bs].unsqueeze(0), weight=dk1, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk2, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk3, stride=1, padding=0)
            output_fpn.append(out)
            dk = k3[1] * 0.7 + SRBP[bs].unsqueeze(0) * 0.3 # 1 584 1 1
            dk1 = dk[:, 0:512, :, :].reshape(8, 64, 1, 1) # 1 512 1 1
            dk2 = dk[:, 512:512+64, :, :].reshape(8, 8, 1, 1)
            dk3 = dk[:, 512+64:, :, :].reshape(1, 8, 1, 1)
            out = F.conv2d(input=D2[bs].unsqueeze(0), weight=dk1, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk2, stride=1, padding=0)
            out = F.conv2d(input=F.relu(out), weight=dk3, stride=1, padding=0)
            output_bkg.append(out)
        output_fpn = torch.cat(output_fpn, dim=0)
        output_bkg = torch.cat(output_bkg, dim=0)
        output_fpn = F.interpolate(output_fpn, size=input.size()[2:], mode='bilinear')
        output_bkg = F.interpolate(output_bkg, size=input.size()[2:], mode='bilinear')
        return out_fore, output_fpn, output_bkg, output_prior, output_priorb


def SR(feature_q, out):
    '''
        feature_q: (B, C, H, W)
        out: (B, 2, H, W)
    '''
    channel = 584
    bs = feature_q.shape[0]
    pred_1 = out.softmax(1)
    pred_1 = pred_1.view(bs, 2, -1)
    pred_fg = pred_1[:, 1]
    pred_bg = pred_1[:, 0]
    fg_ls = []
    bg_ls = []
    for epi in range(bs):
        fg_thres = 0.7
        bg_thres = 0.7
        cur_feat = feature_q[epi].view(channel, -1)
        if (pred_fg[epi] > fg_thres).sum() > 0:
            fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)]
        else:
            fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 3).indices]
        if (pred_bg[epi] > bg_thres).sum() > 0:
            bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)]
        else:
            bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 3).indices]
        fg_proto = fg_feat.mean(-1)
        bg_proto = bg_feat.mean(-1)
        fg_ls.append(fg_proto.unsqueeze(0))
        bg_ls.append(bg_proto.unsqueeze(0))
    new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
    new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

    return new_fg, new_bg
    