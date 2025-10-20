import numpy as np
import os
import time
import torch
from torch.autograd import Variable
from config import polypfive, isic2018, covid, breast, amdsd, btd, ebhi, tnui
from sricl import SRICL
from utils.dataset_rgb_strategy2 import test_get_loader, image_prompt_get_loader
import torch.nn.functional as F
import torchvision.utils as vutils
torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = 'saved_model'
exp_name = 'exp'
args = {
    'snapshot': 'Model_50_gen',
    'save_results': True
}

image_amdsd_root = ".txt"
image_btd_root = ".txt"
image_ebhi_root = ".txt"
image_tnui_root = ".txt"
image_polyp_root = ".txt"
image_covid_root = ".txt"
image_breast_root = ".txt"
image_skin_root = ".txt"

gt_amdsd_root = ".txt"
gt_btd_root = ".txt"
gt_ebhi_root = ".txt"
gt_tnui_root = ".txt"
gt_polyp_root = ".txt"
gt_covid_root = ".txt"
gt_breast_root = ".txt"
gt_skin_root = ".txt"

data_to_test = [{'AMDSD': amdsd}, {'BTD': btd}, {'EBHI': ebhi}, {'TNUI': tnui},
           {'polyp': polypfive}, {'COVID': covid}, {'Breast': breast}, {'ISIC2018': isic2018}]

task_list = ['amdsd','btd','ebhi','tnui','Polyp','COVID','Breast','Skin']


def main():
    t0 = time.time()
    net = SRICL().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']+'.pth'),map_location={'cuda:0': 'cuda:0'}), strict=False)
    net.eval()
    with torch.no_grad():
        for to_test in data_to_test:
            for name, root in to_test.items():
                root1 = os.path.join(root)
                test_image_size = 384
                test_loader = test_get_loader(root1, batchsize=1,trainsize=test_image_size)
                
                batchsize = 4
                if name == 'AMDSD':
                    train_amdsd_loader = image_prompt_get_loader(image_amdsd_root, gt_amdsd_root, batchsize=batchsize, trainsize=test_image_size)
                elif name == 'BTD':
                    train_amdsd_loader = image_prompt_get_loader(image_btd_root, gt_btd_root, batchsize=batchsize, trainsize=test_image_size)
                elif name == 'EBHI':
                    train_amdsd_loader = image_prompt_get_loader(image_ebhi_root, gt_ebhi_root, batchsize=batchsize, trainsize=test_image_size)
                elif name == 'TNUI':
                    train_amdsd_loader = image_prompt_get_loader(image_tnui_root, gt_tnui_root, batchsize=batchsize, trainsize=test_image_size)
                elif name == 'polyp':
                    train_amdsd_loader = image_prompt_get_loader(image_polyp_root, gt_polyp_root, batchsize=batchsize, trainsize=test_image_size)
                elif name == 'COVID':
                    train_amdsd_loader = image_prompt_get_loader(image_covid_root, gt_covid_root, batchsize=batchsize, trainsize=test_image_size)
                elif name == 'Breast':
                    train_amdsd_loader = image_prompt_get_loader(image_breast_root, gt_breast_root, batchsize=batchsize, trainsize=test_image_size)
                elif name == 'ISIC2018':
                    train_amdsd_loader = image_prompt_get_loader(image_skin_root, gt_skin_root, batchsize=batchsize, trainsize=test_image_size)

                # Create save path for predictions
                save_path = './evaluation/predictions/' + name + '/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for i, (query_image, gt, img_name, w_, h_) in enumerate(test_loader):
                    query_image = Variable(query_image).cuda()
                    gt = Variable(gt).cuda()
                    
                    # Binarize query_gt
                    query_gt = gt.unsqueeze(1)
                    query_gt[query_gt != 0] = 1
                    query_gt = query_gt.float()
                    
                    # Get one batch from train_amdsd_loader as support set (randomly sampled each time)
                    for j, (support_images, support_gts) in enumerate(train_amdsd_loader):
                        if j == 0:
                            support_images = Variable(support_images).cuda()
                            support_gts = Variable(support_gts).cuda()
                            # Binarize support_gts
                            support_gts[support_gts != 0] = 1
                            support_gts = support_gts.float()
                            break
                    
                    # Forward pass following training logic
                    output_fpn, output_bkg, _, _ = net(query_image, support_images, support_gts)
                    
                    # Process prediction
                    prediction = torch.cat((output_bkg, output_fpn), dim=1)
                    prediction = F.interpolate(prediction, size=(h_, w_), mode='bilinear').argmax(dim=1, keepdim=True)
                    
                    # Save prediction
                    vutils.save_image(prediction.float() / 255.0, os.path.join(save_path, img_name[0]))

if __name__ == '__main__':
    snapshot = ['Model_50_gen']
    for epo in snapshot:
        args['snapshot'] = epo
        main()
