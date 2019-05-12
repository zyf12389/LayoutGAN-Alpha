import torch
import os
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

class Attention(nn.Module):
    def __init__(self,in_channels,dimension=2,sub_sample=False,bn=True,generate=True):
        super(Attention, self).__init__()
        self.inter_channels=in_channels//2 if in_channels>1 else 1
        self.generate=generate
        if dimension==2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2,2))
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd=nn.Conv1d
            max_pool_layer=nn.MaxPool1d(kernel_size=(2))
            bn=nn.BatchNorm1d

        self.g=conv_nd(in_channels,self.inter_channels,kernel_size=1,stride=1,padding=0)
        if bn:
            self.W=nn.Sequential(conv_nd(self.inter_channels,in_channels,kernel_size=1,stride=1,padding=0),
                                 bn(in_channels))
            nn.init.constant(self.W[1].weight,0)
            nn.init.constant(self.W[1].bias,0)
        else:
            self.W=conv_nd(self.inter_channels,in_channels,kernel_size=1,stride=1,padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta=conv_nd(in_channels,self.inter_channels,kernel_size=1,stride=1,padding=0)
        self.phi=conv_nd(in_channels,self.inter_channels,kernel_size=1,stride=1,padding=0)
        if sub_sample:
            self.g=nn.Sequential(self.g,max_pool_layer)
            self.phi=nn.Sequential(self.phi,max_pool_layer)

    def forward(self,x):
        batch_size=x.size(0)
        g_x=self.g(x).view(batch_size,self.inter_channels,-1)
        g_x=g_x.permute(0,2,1)
        theta_x=self.theta(x).view(batch_size,self.inter_channels,-1)
        theta_x=theta_x.permute(0,2,1)
        phi_x=self.phi(x).view(batch_size,self.inter_channels,-1)
        f=torch.matmul(theta_x,phi_x)
        N=f.size(-1)
        f_div_c=f/N;
        y=torch.matmul(f_div_c,g_x)
        y=y.permute(0,2,1).contiguous()
        y=y.view(batch_size,self.inter_channels,*x.size()[2:])
        W_y=self.W(y)
        if self.generate:
            output=W_y+x
        else:
            output=W_y
        return output

#input [batch,n,feature_dim] linear [N,*,H_in]
# cls_num here is a questionable param. The value of it should be 1?
class Generator(nn.Module):
    def __init__(self,in_channels,num_fea,cls_num=1):
        super(Generator, self).__init__()
        self.cls_num=cls_num

        self.fc1=nn.Linear(in_channels,in_channels*2)
        self.bn1=nn.BatchNorm1d(num_fea)  # bn is needed?
        self.fc2=nn.Linear(in_channels*2,in_channels*2*2)
        self.bn2=nn.BatchNorm1d(num_fea)
        self.fc3=nn.Linear(in_channels*2*2,in_channels*2*2)

        # self.attention_1=Attention(1)
        # self.attention_2=Attention(1)
        # self.attention_3 = Attention(1)
        # self.attention_4 = Attention(1)

        self.attention_1 = Attention(in_channels*2*2,1)
        self.attention_2 = Attention(in_channels*2*2,1)
        self.attention_3 = Attention(in_channels*2*2,1)
        self.attention_4 = Attention(in_channels*2*2,1)

        self.fc4=nn.Linear(in_channels*2*2,in_channels*2)
        self.bn4 = nn.BatchNorm1d(num_fea)
        self.fc5 = nn.Linear(in_channels * 2, in_channels)

        self.fc6=nn.Linear(in_channels,cls_num)
        self.fc7=nn.Linear(in_channels,in_channels-cls_num)

    def forward(self,x):
        x=F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x=F.sigmoid(self.fc3(x))
        #for 2d relation
        # x=x.unsqueeze(1)
        # x = self.attention_1(x)
        # x = self.attention_2(x)
        # x = self.attention_3(x)
        # x = self.attention_4(x)
        # x = x.squeeze(1)

        #for 1d relation
        x=x.permute(0,2,1).contiguous()
        x=self.attention_1(x)
        x=self.attention_2(x)
        x = self.attention_3(x)
        x = self.attention_4(x)
        x=x.permute(0,2,1).contiguous()

        out=F.relu(self.bn4(self.fc4(x)))
        out=F.relu(self.fc5(out))

        cls=F.sigmoid(self.fc6(out))
        geo=F.sigmoid(self.fc7(out))
        output=torch.cat((cls,geo),2)
        return output

class Discriminator(nn.Module):
    def __init__(self,output_channels,num_fea,height,width,cls_num=1):
        super(Discriminator, self).__init__()
        self.cls_num=cls_num
        self.num_elements=num_fea
        self.height=height
        self.width=width
        self.conv1=nn.Conv2d(cls_num,output_channels,kernel_size=3,stride=2)
        self.bn1=nn.BatchNorm2d(output_channels)
        self.conv2=nn.Conv2d(output_channels,output_channels*2,kernel_size=3,stride=2)
        self.bn2=nn.BatchNorm2d(output_channels*2)
        self.conv3=nn.Conv2d(output_channels*2,output_channels*2,kernel_size=3,stride=2)
        self.bn3=nn.BatchNorm2d(output_channels*2)
        self.fc1=nn.Linear(self.cls_num*2*49,output_channels) #2*29*7*7
        self.fc2=nn.Linear(output_channels,1)

    def rectangle_render(self,x):
        # I's size [b,c,h,w]
        # x's size [b,num_ele+5,cls_num+4]
        batch_size=x.size(0)
        # wrong
        # I=torch.zeros((batch_size,self.num_elements,self.height,self.width))
        h_index=torch.arange(0,self.height)
        w_index=torch.arange(0,self.width)
        hh=h_index.repeat(len(w_index))
        ww=w_index.view(-1,1).repeat(1,len(h_index)).view(-1)
        index=torch.stack([ww,hh],dim=-1) #[[0,0],[0,1]...[ww-1,hh-1]]
        index_=index.unsqueeze(0).repeat(batch_size,1,1)
        index_col=index_[:,:,0]
        index_row=index_[:,:,1]
        x_trans=x.permute(0,2,1)
        index_col=index_col.unsqueeze(2)
        index_row=index_row.unsqueeze(2)
        sub_xL=index_col-x_trans[:,self.cls_num,:].unsqueeze(1).long()
        sub_yT=index_row-x_trans[:,self.cls_num+1,:].unsqueeze(1).long()
        sub_xR=index_col-x_trans[:,self.cls_num+2,:].unsqueeze(1).long()
        sub_yB=index_row-x_trans[:,self.cls_num+3,:].unsqueeze(1).long()
        sub_y=x_trans[:,self.cls_num+3,:].unsqueeze(1).long()-index_row
        sub_x=x_trans[:,self.cls_num+2,:].unsqueeze(1).long()-index_col
        tmp1=F.relu(sub_yT)
        tmp1[tmp1>1]=1
        tmp2=F.relu(sub_y)
        tmp2[tmp2>1]=1
        F_0=F.relu(1-torch.abs(sub_xL))*tmp1*tmp2
        F_1=F.relu(1-torch.abs(sub_xR))*tmp1*tmp2
        tmp1 = F.relu(sub_xL)
        tmp1[tmp1 > 1] = 1
        tmp2 = F.relu(sub_x)
        tmp2[tmp2 > 1] = 1
        F_2=F.relu(1-torch.abs(sub_yT))*tmp1*tmp2
        F_3=F.relu(1-torch.abs(sub_yB))*tmp1*tmp2
        # val shape [batch_size,hei*wid,num_elem]
        val,index_ftheta=torch.max(torch.stack((F_0,F_1,F_2,F_3),dim=2),dim=2)

        x_prob=x[:,:,:self.cls_num]
        x_prob=x_prob.unsqueeze(1)#[batch_size,1,num_elem,cls_num]
        F_theta=val.unsqueeze(3).float() #[batch_szie,hei*wid,num_elem,1]
        prod=x_prob*F_theta #[batch_szie,hei*wid,num_elem,cls_num]
        res,index_res=torch.max(prod,2)
        I=res.contiguous().view(batch_size,self.height,self.width,-1).permute(0,1,3,2)
        I=I.permute(0,2,1,3)
        out=F.relu(self.bn1(self.conv1(I)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.relu(self.bn3(self.conv3(out)))
        out=out.view(out.shape[0],-1)
        out=self.fc1(out)
        out=self.fc2(out)
        out=F.sigmoid(out)
        return out
    def forward(self,x):
        output=self.rectangle_render(x)
        return output

if __name__=='__main__':
    x=torch.randn((3,4,5))
    print(x.size())
    model=Generator(5,4)
    output=model(x)
    print(output)
    print(output.size())