import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torchsummary import summary
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
from image_quality_evaluation.model_cnx_classify import cnx_classify
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_mode", type=int, default=1, help="turn on/off train mode")
parser.add_argument("--device", type=str, default='gpu', help="turn on/off train mode")
parser.add_argument("--path_file", type=str, default='', help="path file train")
parser.add_argument("--path_pretrain", type=str, default='', help="path file pretrain model")
parser.add_argument("--path_save_result", type=str, default='result.txt', help="path save result file")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=640,help="size of image height")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout when in train mode")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
opt = parser.parse_args()


train_batchsize = opt.batch_size
train_epochs = opt.n_epochs
pretrain_path = opt.path_pretrain
if pretrain_path == '':
    pretrain_path = None
numb_class = 1
inputsize = opt.img_size
dropout = opt.dropout
channel = opt.channels
gpu = None
if opt.device != 'cpu':
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
train_mode = None
if opt.train_mode != 0:
    train_mode = True

class CNX_Dataset(Dataset):
    def __init__(self, path_fileconfig, transform=None):
        self.path_fileconfig = path_fileconfig
        self.transform = transform
        self.list_path_image = []
        self.list_label = []
        path_dir = os.path.dirname(path_fileconfig)
        with open(path_fileconfig, 'r+') as readf:
            for line  in readf:
                strs = line.split(',')
                self.list_path_image.append(os.path.join(path_dir,strs[0]))
                self.list_label.append(float(strs[1]))

    def __getitem__(self, index):
        imgpath = self.list_path_image[index]
        if channel == 1:
            img = Image.open(imgpath).convert('L')
        else:
            img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.list_label[index]
        torch.from_numpy(np.array([label], dtype=np.float32))
        return img,  torch.from_numpy(np.array([label], dtype=np.float32)), imgpath

    def __len__(self):
        return len(self.list_path_image)

class NumpyListLoader(Dataset):  #no label
    def __init__(self, numpylist, transform=None):
        self.imlist = numpylist
        self.transform = transform

    def __getitem__(self, index):
        imdata = self.imlist[index]
        if channel == 1:
            img = Image.fromarray(imdata).convert('L')
        else:
            img = Image.fromarray(imdata).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, '', ''

    def __len__(self):
        return len(self.imlist)



class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if type(v) is list:
            self.n_count += len(v)
            self.sum += sum(v)
        else:
            self.n_count += 1
            self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, ypre, y):
        loss = torch.sqrt(self.mse(ypre, y))
        return loss


class PCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ypre, y):
        vx = ypre - torch.mean(ypre)
        vy = y - torch.mean(y)

        loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return loss

net = cnx_classify(dropout=dropout, num_classes= numb_class, input_size= inputsize)
if pretrain_path != None:
    state_dict = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict, strict=True)
criterion = RMSELoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
# optimizer = optim.SGD(net.parameters(),lr= 0.0001, weight_decay=1e-6, momentum=0.9, nesterov=True)
if gpu != None:
    net = net.cuda()
    criterion = criterion
dataset = None
train_dataloader = None
eval_dataloader = None
if train_mode == True:
    dataset = CNX_Dataset(path_fileconfig=opt.path_file,
                        transform=transforms.Compose([transforms.Resize((inputsize,inputsize)),
                                                transforms.ToTensor()]))
    train_dataloader = DataLoader(dataset,
                                  shuffle=True,
                                  num_workers=opt.n_cpu,
                                  batch_size=train_batchsize)
else:
    dataset =  CNX_Dataset(path_fileconfig=opt.path_file,
                        transform=transforms.Compose([transforms.Resize((inputsize,inputsize)),
                                                transforms.ToTensor()]))
    eval_dataloader = DataLoader(dataset,
                                  shuffle=False,
                                  num_workers=opt.n_cpu,
                                  batch_size=1)


if train_mode == True:
    train_lost_av = averager()
    summary(net,(3,inputsize,inputsize))
    net.train()
    for epoch in range(1,train_epochs+1):
        print('Epoch {}/{}'.format(epoch, train_epochs))
        for i, data in enumerate(train_dataloader,0):
            img , label , pathimg= data
            if gpu != None:
                img , label =  img.cuda() , label.cuda()
            optimizer.zero_grad()
            valpre = net(img)
            loss= criterion(valpre,label)
            loss.backward()
            optimizer.step()
            train_lost_av.add(loss)
        if epoch%opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), './save_model/new_model_full_rgb_{}_loss_{:.6f}.pth'.format(epoch, train_lost_av.val()))
        print('RMS losss {:.6f}'.format(train_lost_av.val()))
        train_lost_av.reset()
else:
    net.eval()
    count = 0
    loss = 0
    predict_l = []
    path_dir = dataset.list_path_image
    for i, data in enumerate(eval_dataloader, 0):
        img, label , pathimg= data
        if gpu != None:
            img, label = img.cuda(), label.cuda()
        # print(img.size())
        valpre = net(img)
        predict_l.append(valpre.item())
        # count+=1
        # cv2.waitKey()
    print(count)
    content = ''
    for i in range(len(predict_l)):
        content += str(path_dir[i])+', '+str(predict_l[i]) +'\n'

    f = open(opt.path_save_result, 'w+')
    f.write(content)


    # print('loss av ',(loss/count)/5*100)
