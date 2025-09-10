import argparse
import os
import torch
import torch.nn.functional as F
import datetime
from sricl import SRICL
from dataset import SegDataset
from utils import adjust_lr, AvgMeter
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from contextlib import contextmanager
import torch.utils.data as data
import math


def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    dice_coeff = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    dice_loss = 1 - dice_coeff.mean()
    return dice_loss


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return wiou.mean()


def adjust_lr_cosine(optimizer, initial_lr, epoch, total_epochs):
    lr = initial_lr * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_loader(image_root, gt_root, batchsize, trainsize):
    dataset = SegDataset(image_root, gt_root, trainsize)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=12,
                                  pin_memory=True,
                                  sampler=sampler,
                                  drop_last=True
                                  )
    return data_loader, sampler


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0


@contextmanager
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    if is_dist_avail_and_initialized() and rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if is_dist_avail_and_initialized() and rank == 0:
        torch.distributed.barrier()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('-beta1_gen', type=float, default=0.5, help='beta of Adam for generator')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
    return parser.parse_args()


def train():
    opt = get_args()
    print('Generator Learning Rate: {}'.format(opt.lr_gen))

    print('分布式开始初始化...')
    distributed = int(os.environ["WORLD_SIZE"]) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, init_method="env://")
    print('分布式初始化完成!')
    is_master = (distributed and (local_rank == 0)) or (not distributed)

    ## load data
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

    train_amdsd_loader, train_amdsd_sampler = get_loader(image_amdsd_root, gt_amdsd_root, batchsize=opt.batchsize,
                                                     trainsize=opt.trainsize)
    train_btd_loader, train_btd_sampler = get_loader(image_btd_root, gt_btd_root, batchsize=opt.batchsize,
                                                     trainsize=opt.trainsize)
    train_ebhi_loader, train_ebhi_sampler = get_loader(image_ebhi_root, gt_ebhi_root, batchsize=opt.batchsize,
                                                           trainsize=opt.trainsize)
    train_tnui_loader, train_tnui_sampler = get_loader(image_tnui_root, gt_tnui_root,
                                                                     batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_polyp_loader, train_polyp_sampler = get_loader(image_polyp_root, gt_polyp_root, batchsize=opt.batchsize,
                                                         trainsize=opt.trainsize)
    train_covid_loader, train_covid_sampler = get_loader(image_covid_root, gt_covid_root, batchsize=opt.batchsize,
                                                         trainsize=opt.trainsize)
    train_breast_loader, train_breast_sampler = get_loader(image_breast_root, gt_breast_root, batchsize=opt.batchsize,
                                                           trainsize=opt.trainsize)
    train_skin_loader, train_skin_sampler = get_loader(image_skin_root, gt_skin_root, batchsize=opt.batchsize,
                                                       trainsize=opt.trainsize)
    total_step = len(train_amdsd_loader)

    size_rates = [1]  # multi-scale training
    use_fp16 = True# False

    save_path = './saved_model/'
    with torch_distributed_zero_first(rank=local_rank):
        os.makedirs(save_path, exist_ok=True)

    log_path = os.path.join(save_path, str(datetime.datetime.now()) + '.txt')
    if is_master:
        open(log_path, 'w')

    print("开始初始化模型，优化器...")
    generator = SRICL(model='base')
    generator.cuda()

    generator_optimizer = torch.optim.Adam(generator.parameters(), opt.lr_gen)
    scaler = amp.GradScaler(enabled=use_fp16)

    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    print("Start Training...")
    for epoch in range(1, opt.epoch + 1):
        train_amdsd_sampler.set_epoch(epoch)
        train_btd_sampler.set_epoch(epoch)
        train_ebhi_sampler.set_epoch(epoch)
        train_tnui_sampler.set_epoch(epoch)
        train_polyp_sampler.set_epoch(epoch)
        train_covid_sampler.set_epoch(epoch)
        train_breast_sampler.set_epoch(epoch)
        train_skin_sampler.set_epoch(epoch)
        generator.train()
        loss_record = AvgMeter()
        if is_master:
            print('Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

        for i, (
                (image_amdsd, gt_amdsd),
                (image_btd, gt_btd),
                (image_ebhi, gt_ebhi),
                (image_tnui, gt_tnui),
                (image_polyp, gt_polyp),
                (image_covid, gt_covid),
                (image_breast, gt_breast),
                (image_skin, gt_skin),
        ) in enumerate(zip(
            train_amdsd_loader,
            train_btd_loader,
            train_ebhi_loader,
            train_tnui_loader,
            train_polyp_loader,
            train_covid_loader,
            train_breast_loader,
            train_skin_loader,
        ), start=1):

            images = [image_amdsd, image_btd, image_ebhi, image_tnui, image_polyp, image_covid, image_breast,
                      image_skin]
            gts = [gt_amdsd, gt_btd, gt_ebhi, gt_tnui, gt_polyp, gt_covid, gt_breast, gt_skin]
            num_tasks = len(images)

            recurrent_set = 4
            batch_set = gt_amdsd.shape[0] // recurrent_set

            for rate in size_rates:
                for curr_idx in range(recurrent_set):
                    avg_grads = [torch.zeros_like(param, dtype=torch.float32).cuda() for param in generator.parameters()]
                    for task_idx in range(num_tasks):
                        image = images[task_idx].cuda().unsqueeze(1)
                        gt = gts[task_idx].cuda().unsqueeze(1)
                        query_image = image[curr_idx * batch_set:(curr_idx + 1) * batch_set]
                        query_gt = gt[curr_idx * batch_set:(curr_idx + 1) * batch_set]

                        support_images = torch.cat([image[:curr_idx * batch_set], image[(curr_idx + 1) * batch_set:]],
                                                   dim=0)
                        support_gts = torch.cat([gt[:curr_idx * batch_set], gt[(curr_idx + 1) * batch_set:]], dim=0)

                        trainsize = int(round(opt.trainsize * rate / 32) * 32)
                        if rate != 1:
                            query_image = F.upsample(query_image, size=(trainsize, trainsize), mode='bilinear',
                                                     align_corners=True)
                            query_gt = F.upsample(query_gt, size=(trainsize, trainsize), mode='bilinear',
                                                  align_corners=True)

                        with amp.autocast(enabled=use_fp16):
                            support_images = support_images.permute(1, 0, 2, 3, 4)
                            support_gts = support_gts.permute(1, 0, 2, 3, 4)
                            support_gts[support_gts != 0] = 1
                            support_gts = support_gts.float()
                            query_gt[query_gt != 0] = 1
                            query_gt = query_gt.float()

                            query_image = query_image.permute(1, 0, 2, 3, 4)
                            query_image = query_image.reshape(-1, 3, image_amdsd.shape[2], image_amdsd.shape[3])
                            query_gt = query_gt.permute(1, 0, 2, 3, 4)
                            query_gt = query_gt.reshape(-1, 1, image_amdsd.shape[2], image_amdsd.shape[3])

                            output_fpn, output_bkg, output_fpn_0, output_bkg_0 = generator(query_image, support_images, support_gts)
                            loss = structure_loss(output_fpn[0:batch_set], query_gt[0:batch_set]) + structure_loss(output_bkg[0:batch_set], 1. - query_gt[0:batch_set]) \
                                    + dice_loss(output_fpn[0:batch_set], query_gt[0:batch_set]) + dice_loss(output_bkg[0:batch_set], 1. - query_gt[0:batch_set]) \
                                    + structure_loss(output_fpn_0[0:batch_set], query_gt[0:batch_set]) + structure_loss(output_bkg_0[0:batch_set], 1. - query_gt[0:batch_set]) \
                                    + dice_loss(output_fpn_0[0:batch_set], query_gt[0:batch_set]) + dice_loss(output_bkg_0[0:batch_set], 1. - query_gt[0:batch_set])

                        generator_optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        
                        for avg_grad, param in zip(avg_grads, generator.parameters()):
                            if param.grad is not None:
                                avg_grad.add_(param.grad)

                    for param, avg_grad in zip(generator.parameters(), avg_grads):
                        if param.grad is not None:
                            param.grad = (avg_grad / (8 * len(size_rates))).to(param.dtype)

                    scaler.step(generator_optimizer)
                    scaler.update()

                if rate == 1:
                    loss_record.update(loss.data, opt.batchsize)

            if is_master:
                if i % 10 == 0 or i == total_step:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                          format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

                    log = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                           format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
                    open(log_path, 'a').write(log + '\n')

        adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

        if is_master:
            eval(generator, epoch, opt.epoch, log_path)
            if epoch % 50 == 0:
                torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
            if epoch % opt.epoch == 0:
                torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')

if __name__ == '__main__':
    train()
