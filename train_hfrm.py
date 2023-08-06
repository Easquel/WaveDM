import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image
import pdb
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models.vgg as vgg
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from models.model_dense import *      #unet...................................
from models.arch import HFRM

from datasets.dataset import * 

import torch.nn as nn
import torch.nn.functional as F
import torch

def BatchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean(dim=(1,2,3)).sqrt()
    ps = 20 * torch.log10(1/rmse)
    return ps

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
# moxing environment
# import moxing as mox
# mox.file.shift('os','mox')

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input-output))
        
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  #pdb.set_trace()    #15*32*32
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)

class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",        #1_2 to 5_2
        }
        
    def forward(self, x):
        output = {}
        #import pdb
        #pdb.set_trace()
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        
        return output
        
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
        
        
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="raindrop", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')      
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')    # image put in the network
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')

parser.add_argument('--data_url', type=str, default="", help='name of the dataset')
parser.add_argument('--init_method', type=str, default="", help='name of the dataset')
parser.add_argument('--train_url', type=str, default="", help='name of the dataset')

opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()#  smoothl1loss()
tvloss = TVLoss()
lossmse = torch.nn.MSELoss()

# if use GAN loss
lambda_pixel = 100
patch = (1, opt.img_height//2**4, opt.img_width//2**4)   

# Initialize
img_channel = 3
#big one
# dim = 64
# enc_blks = [2, 2, 2, 8]
# middle_blk_num = 12
# dec_blks = [2, 2, 2, 2]
dim = 32
enc_blks = [2, 2, 2, 4]
middle_blk_num = 6
dec_blks = [2, 2, 2, 2]
generator = HFRM(in_channel=img_channel, dim=dim, mid_blk_num=middle_blk_num,enc_blk_nums=enc_blks,dec_blk_nums=dec_blks)
pytorch_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("Total_params_model: {}M".format(pytorch_total_params/1000000.0))

if cuda:
    generator = generator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    lossnet = LossNetwork().float().cuda()

    #discriminator=nn.DataParallel(discriminator,device_ids=[0,1])

if opt.epoch != 0:
    generator.load_state_dict(torch.load('./saved_models/raindrop/best.pth' ),strict=True)#%  opt.epoch))
else:
    # Initialize weights
    generator.apply(weights_init_normal)

generator=nn.DataParallel(generator)
device = torch.device("cuda:0")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))

mytransform = transforms.Compose([    
     transforms.ToTensor(),   
     #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
# change the root to your own data path
data_root = '/data1/jchuang/weather/raindrop/train'
myfolder = myImageFloder(root = data_root,  transform = mytransform,crop=False,resize=False,crop_size=480,resize_size=480)
dataloader = DataLoader(myfolder, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True)
print('data loader finish！')


def get_mask(dg_img,img):
    mask = np.fabs(dg_img.cpu()-img.cpu())
    mask[mask<(20.0/255.0)] = 0.0
    mask = mask.cuda()
    return mask

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(epoch , i ,real_A,real_B,fake_B):
    data,pred,label = real_A *255 , fake_B *255, real_B *255
    data = data.cpu()
    pred = pred.cpu()
    label = label.cpu()
    #pdb.set_trace()
    pred = torch.clamp(pred.detach(),0,255)
    data,pred,label = data.int(),pred.int(),label.int()
    h,w = pred.shape[-2],pred.shape[-1]
    img = np.zeros((h,1*3*w,3))
    #pdb.set_trace()
    for idx in range(0,1):
        row = idx*h
        tmplist = [data[idx],pred[idx],label[idx]]
        for k in range(3):
            col = k*w
            tmp = np.transpose(tmplist[k],(1,2,0))
            img[row:row+h,col:col+w]=np.array(tmp)
    #pdb.set_trace()
    img = img.astype(np.uint8)
    img= Image.fromarray(img)
    img.save("./train_result/%03d_%06d.png"%(epoch,i))
    
# ----------
#  Training
# ----------
EPS = 1e-12
prev_time = time.time()
step = 0
best_psnr = 31
for epoch in range(opt.epoch, opt.n_epochs):

    epoch_psnr = []
    for i, batch in enumerate(tqdm(dataloader), 0):
        step = step+1
        
        # set lr rate
        current_lr = 0.0002*(1/2)**(step/100000)
        for param_group in optimizer_G.param_groups:
            param_group["lr"] = current_lr
            
        # Model inputs
        img_train = batch
        real_A, real_B = Variable(img_train[0].cuda()), Variable(img_train[1].cuda())
        #pdb.set_trace()

        batch_size = real_B.size(0)
        
        if epoch >-1 :

            optimizer_G.zero_grad()

            fake_B = generator(real_A)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)   #.................................

            # preceptual loss
            # loss_fake_B = lossnet(fake_B*255)
            # loss_real_B = lossnet(real_B*255)
            p0=compute_l1_loss(fake_B*255,real_B*255)*2
            # p1=compute_l1_loss(loss_fake_B['relu1'],loss_real_B['relu1'])/2.6
            # p2=compute_l1_loss(loss_fake_B['relu2'],loss_real_B['relu2'])/4.8
            #p3=compute_l1_loss(loss_fake_B['relu3'],loss_real_B['relu3'])/3.7
            #p4=compute_l1_loss(loss_fake_B['relu4'],loss_real_B['relu4'])/5.6
            #p5=compute_l1_loss(loss_fake_B['relu5'],loss_real_B['relu5'])/5.6     #   *10/1.5  
            loss_p = p0 #+p1+p2   #+p3+p4+p5

            loss_G = 1*loss_p  # +  loss_tv  loss_pixel
            
            loss_G.backward()

            optimizer_G.step()

            psnr = BatchPSNR(fake_B,real_B)
            epoch_psnr.append(psnr.mean().item())
            print("PSNR this: %f", psnr.mean().item())

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            if i%100==0:
                print("G loss: %f",loss_G.item())
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                         loss_G.item(),
                                                        loss_pixel.item(),
                                                        time_left)) 
            
            if i % 1000==0:
                sample_images(epoch , i ,real_A,real_B,fake_B)
                
                
        else:
            pass


    print("epoch PSNR: %f, best psnr:%f"%(np.mean(epoch_psnr),best_psnr))
    if np.mean(epoch_psnr) > best_psnr:
        best_psnr = np.mean(epoch_psnr)
        torch.save(generator.module.state_dict(), './saved_models/%s/best.pth' % opt.dataset_name)

    torch.save(generator.module.state_dict(),'./saved_models/%s/lastest.pth'%opt.dataset_name)
    if epoch+1 % 20==0:
      torch.save(generator.module.state_dict(), './saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
      
      

