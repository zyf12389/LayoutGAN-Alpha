import sys,os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,p)

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from dataset import Dataset
from layoutgan import Generator
from layoutgan import Discriminator
import numpy as np
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default="bedroom", metavar='S')
parser.add_argument('--save-resdir', type=str, default="train/bedroom", metavar='S')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--train-size', type=int, default=6400, metavar='N')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--ablation', type=str, default=None, metavar='S')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', type=str, default='LayoutGAN/result',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)
with open(f"{p}/data/{opt.data_dir}/final_categories_frequency","r") as f:
    lines = f.readlines()
num_categories = len(lines) - 2

# ---for wirte log----#
# logfile=open('./{}/log.txt'.format(opt.outf),'w')
logfile = open('./result/log.txt'.format(opt.outf), 'w')
writer = SummaryWriter()


def Log(msg):
    print(msg)
    logfile.write(msg + '\n')
    logfile.flush()


try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = Dataset(data_root_dir=utils.get_data_root_dir(),
                  data_dir=opt.data_dir,
                  scene_indices=(0, opt.train_size),
                  num_per_epoch=1)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


def real_loss(D_out, smooth=False):
    labels = None
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    crit = nn.BCEWithLogitsLoss()
    loss = crit(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    crit = nn.BCEWithLogitsLoss()
    loss = crit(D_out.squeeze(), labels)
    return loss


element_num = num_categories + 5

netG = Generator(num_categories + 4, element_num, num_categories)
netD = Discriminator(num_categories, element_num, opt.imageSize, opt.imageSize, num_categories)
print(netG)
print(netD)

#netG = torch.nn.DataParallel(netG).cuda()
#netD = torch.nn.DataParallel(netD).cuda()


d_optimizer = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
g_optimizer = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

cls_num = num_categories
geo_num = 4
print("cls_num=", cls_num, "\ngeo_num=", geo_num)

for epoch in range(opt.niter):
    netD.train()

    print("\nepoch ", epoch,
          " netD train finish")

    netG.train()

    print("epoch ", epoch,
          " netG train finish")

    for batch_i, real_images in enumerate(dataloader):
        batch_size = real_images.size(0)

        # Train Discriminator.
        d_optimizer.zero_grad()

        D_real = netD(real_images)
        d_real_loss = real_loss(D_real)

        # !Random layout input generation have logic error, should be fixed.
        zlist = []
        for i in range(batch_size):
            ##############zc修改
            # cls_z = np.ones((element_num, num_categories))
            cls_z = np.random.uniform(size=(element_num, num_categories))
            geo_z = np.random.normal(0, 1, size=(element_num, geo_num))

            z = torch.FloatTensor(np.concatenate((cls_z, geo_z), axis=1))
            zlist.append(z)

        fake_images = netG(torch.stack(zlist))

        D_fake = netD(fake_images)
        d_fake_loss = fake_loss(D_fake)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        # !Random layout input generation have logic error, should be fixed.
        zlist2 = []
        for i in range(batch_size):
            ##############zc修改
            # cls_z = np.ones((element_num, cls_num))
            cls_z = np.random.uniform(size=(element_num, cls_num))
            geo_z = np.random.normal(0, 1, size=(element_num, geo_num))

            z = torch.FloatTensor(np.concatenate((cls_z, geo_z), axis=1))
            zlist2.append(z)

        fake_images2 = netG(torch.stack(zlist2))
        D_fake = netD(fake_images2)
        g_loss = real_loss(D_fake)
        writer.add_scalar('data/D_LOSS', d_loss, epoch)
        writer.add_scalar('data/G_LOSS', g_loss, epoch)
        writer.add_scalars('data/D_G_LOSS', {'D_LOSS': d_loss,
                                             'G_LOSS': g_loss}, epoch)
        print_msg = '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % \
                    (epoch + 1, opt.niter, batch_i, len(dataloader), d_loss.item(), g_loss.item())
        Log(print_msg)

    if epoch % 5 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
