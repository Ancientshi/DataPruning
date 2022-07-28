'''Train CIFAR10 with PyTorch.'''
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import random
import time
from tqdm import tqdm
from math import cos, pi
import numpy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torchsummary import summary

from models import *
from utils import progress_bar,init_params

from dataloader import *
from config import *

from torch.optim.lr_scheduler import StepLR, ExponentialLR,CosineAnnealingLR
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

trainer_args=get_args()

seed=trainer_args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def adjust_learning_rate(optimizer, warmup_epoch,current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    if current_epoch<7:
        lr =0.12
    elif current_epoch>=7 and current_epoch<14:
        lr = 0.08
    elif current_epoch>=14 and current_epoch<28:
        lr = 0.08
    else:
            lr = lr_min + (0.1 - 0.05) * (
                        1 + cos(pi * (current_epoch - 28) / (max_epoch - 28))) / 2

    # if current_epoch < warmup_epoch:
    #     lr = lr_max * current_epoch / warmup_epoch
    # else:
    #     lr = lr_min + (lr_max - lr_min) * (
    #                 1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

transform_train = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_pruning = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class Trainer(object):
    getBatch_init_flag=False
    batch_dict={}
    def __init__(self,task_args=None):
        self.args = trainer_args
        if task_args!=None:
            self.args.save_net=task_args.save_net
            self.args.lr=task_args.lr
            self.args.train_batch_size=task_args.train_batch_size
            self.args.net_name=task_args.net_name
            self.mode=task_args.mode
        else:
            self.mode='SGD'

        # Data
        print('==> Preparing data..')
        self.train_data  = CIFAR10_DataPruning(
            root=self.args.data_path, 
            train=True, 
            download=False, 
            transform=transform_train,
            datapruning=False
        )
        self.trainloader = DataLoader(self.train_data, batch_size=self.args.train_batch_size,shuffle=False, num_workers=1,worker_init_fn=seed_worker,generator=g)

        #train_batch_size
        self.B=self.args.train_batch_size
        ##一个epoch的迭代次数
        self.C=int(len(self.train_data)/self.B)
        #总的epoch次数
        self.T=self.args.epoch
        #k为要删除的数据
        self.k=-1
        #删除的Xk的批次
        self.q=-1

        if self.mode=='SGD':
            self.args.save_path='/home/yunxshi/Data/workspace/DataPruning/TrainPhase/'
        elif self.mode=='SGD_k':
            self.k=task_args.k
            #k在批次中的index，8888%200=88，索引是87
            self.index_inBatch=(self.k-1)%self.B
            #处于一个epoch中的第几个batch，(8888-1)/200=44.435,所以是第44+1个batch,由于batch in [0,I-1],所以44+1-1=44
            self.q=math.floor((self.k-1)/self.B)
            self.args.save_path='/home/yunxshi/Data/workspace/DataPruning/TrainPhase_k/'
        elif self.mode=='Infer_k':
            self.k=task_args.k
            #k在批次中的index，8888%200=88，索引是87
            self.index_inBatch=(self.k-1)%self.B
            #处于一个epoch中的第几个batch，(8888-1)/200=44.435,所以是第44+1个batch,由于batch in [0,I-1],所以44+1-1=44
            self.q=math.floor((self.k-1)/self.B)
            self.args.save_path='/home/yunxshi/Data/workspace/DataPruning/InferPhase/'
        elif self.mode=='Infer_all':
            self.args.save_path='/home/yunxshi/Data/workspace/DataPruning/InferPhase/'
        elif self.mode=='SGD_dataPruning':
            self.args.save_path='/home/yunxshi/Data/workspace/DataPruning/TrainPhase_dataPruning/'
            self.pos=task_args.pos
            self.lossChangePath=task_args.lossChangePath
            self.pruningSize=task_args.pruningSize
            self.train_data_dataPruning  = CIFAR10_DataPruning(
                root=self.args.data_path, 
                train=True, 
                download=False, 
                transform=transform_train_pruning,
                datapruning=True,
                lossChangePath=self.lossChangePath,
                pruningSize=self.pruningSize,
                pos=self.pos
            )
            self.trainloader_dataPruning = DataLoader(self.train_data_dataPruning, batch_size=self.args.train_batch_size,shuffle=True, num_workers=1,worker_init_fn=seed_worker,generator=g)
        elif self.mode=='SGD_randomPruning':
            self.args.save_path='/home/yunxshi/Data/workspace/DataPruning/TrainPhase_randomPruning/'
            self.pruningSize=task_args.pruningSize
            self.train_data_randomPruning  = CIFAR10_DataPruning(
                root=self.args.data_path, 
                train=True, 
                download=False, 
                transform=transform_train_pruning,
                datapruning=True,
                lossChangePath='',
                pruningSize=self.pruningSize
            )
            self.trainloader_randomPruning = DataLoader(self.train_data_randomPruning, batch_size=self.args.train_batch_size,shuffle=True, num_workers=1,worker_init_fn=seed_worker,generator=g)

        self.dirName='''%s_%s_%s'''%(self.args.net_name,self.args.train_batch_size,self.args.lr)
        self.dirName=self.args.save_path+self.dirName
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0  # best test accuracy
        



        self.test_data  = CIFAR10_DataPruning(
            root=self.args.data_path, 
            train=False, 
            download=False, 
            transform=transform_test,
            datapruning=False
        )
        self.testloader = DataLoader(self.test_data, batch_size=self.args.test_batch_size,shuffle=False, num_workers=1,worker_init_fn=seed_worker,generator=g)



        self.setModel()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.scheduler =optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)


        self.summary()

        self.use_getBatch_init=True
        self.mean_BatchGrad_dict={}
        self.alpha_dict={}
        self.net_dict={}
        
        
    def setModel(self):
        # Model
        netName=self.args.net_name
        print('Using %s, ==> Building model...'%(netName))
        if netName=='GoogLeNet':
            net = GoogLeNet()
        elif netName=='MyNet':
            net = MyNet()
        elif netName=='LeNet':
            net = LeNet()
        elif netName=='VGG11':
            net =VGG('VGG11')
        elif netName=='ResNet50':
            net =ResNet50()
            #net=resnet50()
            #net= torchvision.models.resnet50(pretrained=False)
            
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = PreActResNet18()
        
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()

        self.net = net.to(self.device)
        init_params(self.net)

    def getGrad(self,loss=None):
        '''
        获取模型梯度，若loss不为None，会backward之后再取梯度
        '''
        if loss is None:
            #说明调用getGrad之时没有给Loss，那就返回梯度就行，不用zero_grad()
            pass
        else:
            #梯度清零
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  
        p_grad_list=[]
        for param in self.net.parameters():
            p_grad=param.grad.reshape(-1)
            p_grad_list.append(p_grad)
        grad=torch.cat(p_grad_list,-1)
        return grad.reshape(1,-1)
        
    # def getBatchGrad(self,batch_loss):
    #     '''
    #     batch_loss为一个批次的loss，
    #     会返回一个批次的梯度
    #     '''
    #     batchGrad=torch.cuda.FloatTensor()
    #     for i in range(self.args.train_batch_size):
    #         grad=self.getGrad(batch_loss[i])
    #         #梯度清零
    #         self.optimizer.zero_grad()
    #         if batchGrad.shape[0]==0:
    #             batchGrad=grad
    #         else:
    #             batchGrad=torch.cat((batchGrad, grad),0)
    #     return batchGrad

    def getLoss(self,trainset_index):
        '''
        计算trainset中index下的loss
        '''
        self.net.eval()
        index_of_Xk=trainset_index
        inputs, targets=self.train_data[index_of_Xk]
        inputs=inputs.unsqueeze(0)
        targets=torch.tensor([targets])
        inputs, targets=inputs.to(self.device), targets.to(self.device)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def prepareInfer(self,k=1,p=0):
        #train_batch_size
        self.B=self.args.train_batch_size
        #总的epoch次数
        self.T=self.args.epoch
        #一个epoch的迭代次数
        self.C=int(len(self.train_data)/self.B)
        #一共有10000条数据，k in [1,10000],Xk是要删除的数据
        self.k=k
        #k在批次中的index，8888%200=88，索引是87
        self.index_inBatch=(self.k-1)%self.B
        #在epoch为p删除掉Xk,t in [0,T-1],p默认为0
        self.p=p
        #处于一个epoch中的第几个batch，(8888-1)/200=44.435,所以是第44+1个batch,由于batch in [0,I-1],所以44+1-1=44
        self.q=math.floor((self.k-1)/self.B)
        #在第pi次batch，删除了Xk
        self.pi=int(self.p*self.C+self.q)
        #print('p,q,pi,',self.p,self.q,self.pi)


        print('一共有%s个Epoch，\n每个Epoch有%s个batch，\n一个batch有%s条数据\n'%(self.T,self.C,self.B))
        print('在第%s个Epoch删掉训练数据中的第%s个数据，\n这条数据处于每个Epoch中的第%s个batch中，\n他在第%s个batch中被删去，\n他在这个batch中的index为%s\n'%(self.p,self.k,self.q,self.pi,self.index_inBatch))
    
    def calTau(self,w_i):
        '''
        传入的i是W的上标
        '''
        try:
            alpha=self.alpha_dict[w_i]
            net=self.net_dict[w_i-1]
        except:
            alpha=self.getCheckpoint(w_i)['lr']
            self.alpha_dict[w_i]=alpha
            net=self.getCheckpoint(w_i-1)['net']
            self.net_dict[w_i-1]=net
            self.net.load_state_dict(net)  

        B=self.B
        inputs, targets =self.getBatch(self.pi)
        
        #若是在All模式下，每条数据都要求在w_i时的梯度，没法优化
        input_Xk, target_Xk=inputs[self.index_inBatch],targets[self.index_inBatch]
        input_Xk, target_Xk = input_Xk.to(self.device), target_Xk.to(self.device)
        input_Xk=input_Xk.unsqueeze(0)
        target_Xk =target_Xk.unsqueeze(0)
        self.net.eval()
        output_Xk = self.net(input_Xk)
        loss_Xk= self.criterion(output_Xk, target_Xk)[0]
        grad_Xk=self.getGrad(loss_Xk)

        #计算在pi那个批次下，在w_i时的梯度均值，这个会多次计算到，用备忘录记下
        try:
            mean_BatchGrad=self.mean_BatchGrad_dict[self.pi]
        except:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.net.eval()
            outputs = self.net(inputs)
            loss= self.criterion(outputs, targets)
            loss_mean=torch.mean(loss,0)
            mean_BatchGrad=self.getGrad(loss_mean)
            self.mean_BatchGrad_dict[self.pi]=mean_BatchGrad
        
        tau=(alpha/(B-1))*(grad_Xk-mean_BatchGrad)
        return tau

    def calJu(self,u,i):
        checkpoint=self.getCheckpoint(i-1)
        self.net.load_state_dict(checkpoint['net'])
        inputs, targets =self.getBatch(i)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.net.eval()
        outputs = self.net(inputs)
        loss= self.criterion(outputs, targets)
        mean_loss=torch.mean(loss,0)
        Xk_loss=loss[self.index_inBatch]
        grad1 = torch.autograd.grad(Xk_loss-mean_loss, self.net.parameters(), 
                            create_graph=True, 
                            retain_graph=True) # 为计算二阶导保持计算图
        m_list=[]
        for m in grad1:
            m=m.reshape(-1)
            m_list.append(m)
        grad1_=torch.cat(m_list,-1)
        prod=torch.dot(u.reshape(-1),grad1_)
        grad2 = torch.autograd.grad(prod, self.net.parameters(), 
                        create_graph=False,
                        retain_graph=True) 
        m_list=[]
        for m in grad2:
            m=m.reshape(-1)
            m_list.append(m)
        grad2_=torch.cat(m_list,-1)                
        Ju=grad2_/self.B
        return Ju

    def calJu_forAll(self,u,i):
        checkpoint=self.getCheckpoint(i-1)
        self.net.load_state_dict(checkpoint['net'])
        inputs, targets =self.getBatch(i)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.net.eval()
        outputs = self.net(inputs)
        loss= self.criterion(outputs, targets)
        mean_loss=torch.mean(loss,0)
        grad1 = torch.autograd.grad(((self.B-1)/self.B)*mean_loss, self.net.parameters(), 
                            create_graph=True, 
                            retain_graph=True) # 为计算二阶导保持计算图

        m_list=[]
        for m in grad1:
            m=m.reshape(-1)
            m_list.append(m)
        grad1_=torch.cat(m_list,-1)
        prod=torch.dot(u.reshape(-1),grad1_)
        grad2 = torch.autograd.grad(prod, self.net.parameters(), 
                        create_graph=False,
                        retain_graph=True) 
        m_list=[]
        for m in grad2:
            m=m.reshape(-1)
            m_list.append(m)
        grad2_=torch.cat(m_list,-1)                
        Ju=grad2_/self.B
        return Ju

    def calHu(self,u,i):
        '''
        i是上标
        '''
        checkpoint=self.getCheckpoint(i-1)
        self.net.load_state_dict(checkpoint['net'])
        inputs, targets =self.getBatch(i)
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.net.eval()
        outputs = self.net(inputs)
        loss= self.criterion(outputs, targets)

        loss=torch.sum(loss,0)

        grad1 = torch.autograd.grad(loss, self.net.parameters(), 
                            create_graph=True, 
                            retain_graph=True) # 为计算二阶导保持计算图
        m_list=[]
        for m in grad1:
            m=m.reshape(-1)
            m_list.append(m)
        grad1_=torch.cat(m_list,-1)
        prod=torch.dot(u.reshape(-1),grad1_)
        grad2 = torch.autograd.grad(prod, self.net.parameters(), 
                        create_graph=False,
                        retain_graph=True) 
        m_list=[]
        for m in grad2:
            m=m.reshape(-1)
            m_list.append(m)
        grad2_=torch.cat(m_list,-1)                
        Hu=grad2_/self.B
        return Hu

    def calUZ(self,u,j):
        '''
        j是Z的上标
        '''
        #print('正在计算u*Z^%s'%j)
        t,c=self.i2tc(j)
        alpha=self.getCheckpoint(j)['lr']
        B=self.B
        if j>=self.pi+1 and c==self.q+1:
            Hu=self.calHu(u,j)
            if self.use_apx_ju:
                #print('使用apx ju')
                Ju=self.calJu_forAll(u,j)  #这个是方便计算所有k形式的
            else:
                Ju=self.calJu(u,j) #这个是标准的
            u=u+alpha*(Ju-Hu)
        elif j>=self.pi+1 and c!=self.q+1:
            Hu=self.calHu(u,j)
            u=u-alpha*Hu
        return u

    def calUZ_forAll(self,u,j,q=0):
        '''
        j是Z的上标,j最小为pi+1
        '''
        #print('正在计算u*Z^%s'%j)
        t,c=self.i2tc(j)
        alpha=self.getCheckpoint(j)['lr']
        B=self.B
        if j>=q+1 and c==q+1:
            Hu=self.calHu(u,j)
            Ju=self.calJu_forAll(u,j)  #这个是方便计算所有k形式的
            u=u+alpha*(Ju-Hu)
        elif j>=q+1 and c!=q+1:
            Hu=self.calHu(u,j)
            u=u-alpha*Hu
        return u


    def initU(self,w_i=-1):
        '''
        这里的i表示的是W^i
        '''
        checkpoint=self.getCheckpoint(w_i-1)
        self.net.load_state_dict(checkpoint['net'])

        self.net.eval()
        temp_loss_list=[]
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net(inputs)
            temp_loss = self.criterion(outputs, targets)
            temp_loss_list.append(temp_loss)
        loss=torch.cat(temp_loss_list,0)
        loss=torch.mean(loss,dim =0)
        return self.getGrad(loss)

    def calU_forAll(self,w_i):
        uDict={}
        #正常是从i-1到pi+1,这里一直到0，都是闭区间，j表示Z的上标
        u_init=self.initU(w_i)
        for q in range(0,self.C):
            #print('假设q为%s：'%q)
            u=u_init[:]
            if w_i==q+1:
                #print('得到u: %s_%s'%(0,q))
                uDict['%s_%s'%(0,q)]=u
            for j in tqdm(range(w_i-1,q,-1)):
                #第一个是上标，第二个是下标，第三个是假设的q
                u=self.calUZ_forAll(u,j,q)
                t_j,c_j=self.i2tc(j-1)
                if (j-q-1)%self.C==0:
                    #print('得到u: %s_%s'%(t_j,q))
                    uDict['%s_%s'%(t_j,q)]=u
        self.uDict=uDict

    def calU(self,w_i):
        '''
        从i-1到pi+1,j表示Z的上标
        '''
        uDict={}
        u_init=self.initU(w_i)
        u=u_init[:]
        if w_i==self.pi+1:
            uDict[0]=u

        for j in tqdm(range(w_i-1,self.pi,-1)):
            u=self.calUZ(u,j)
            t_j,c_j=self.i2tc(j)
            if (j-self.pi-1)%self.C==0:
                uDict[t_j]=u
        self.uDict=uDict


    def calLossChangeAll(self,w_i=30):
        kList=self.kList
        if len(kList)==0:
            kList=range(1,len(self.train_data)+1)
        w_i=int(self.T*self.I if w_i==-1 else w_i)
        t,c=self.i2tc(w_i)
        self.calU_forAll(w_i)
        result_dict={}
        for k in tqdm(kList):
            self.prepareInfer(k,p=0)
            if w_i<=self.pi:
                result_dict[k]=0.0
            elif w_i==self.pi+1:
                u=self.uDict['%s_%s'%(0,self.q)]
                tau=self.calTau(self.pi)
                prod=torch.dot(u.squeeze(0),tau.squeeze(0))
                result=prod.to('cpu').numpy().tolist()
                result_dict[k]=result
            elif w_i>=self.pi+2:
                u_list=[]
                tau_list=[]
                for v in range(0,t+1):
                    if v*self.C+self.pi+1<=w_i-1:
                        #计算u^{i}_{vC+\pi+1}
                        u=self.uDict['%s_%s'%(v,self.q)]
                        #计算\tau
                        tau=self.calTau(v*self.C+self.pi)
                        u_list.append(u)
                        tau_list.append(tau)
                u_matrix=torch.stack(u_list,dim=0) 
                tau_matrix=torch.stack(tau_list,dim=0) 
                result=torch.mul(u_matrix,tau_matrix).sum().item()
                result_dict[k]=result
        return result_dict

    def calLossChange(self,w_i=-1):
        '''
        w_i表示的是W的上标,i应该为[1,TC],这样才能与real的[0,TC-1] 进行对比实验
        '''
        assert 1<=w_i<=self.T*self.C
        if w_i<=self.pi:
            return 0.0
        elif w_i==self.pi+1:
            self.calU(w_i)
            u=self.uDict[0]
            tau=self.calTau(self.pi)
            prod=torch.dot(u.squeeze(0),tau.squeeze(0))
            return prod.to('cpu').numpy().tolist()
        elif w_i>=self.pi+2:
            t,c=self.i2tc(w_i)
            self.calU(w_i)
            deltaLoss=torch.zeros(1).to('cuda:0')
            u_list=[]
            tau_list=[]
            for v in range(0,t+1):
                if v*self.C+self.pi+1<=w_i-1:
                    #计算u^{i}_{vC+\pi+1}
                    u=self.uDict[v]
                    #计算\tau
                    tau=self.calTau(v*self.C+self.pi)
                    u_list.append(u)
                    tau_list.append(tau)
            u_matrix=torch.stack(u_list,dim=0) 
            tau_matrix=torch.stack(tau_list,dim=0) 
            result=torch.mul(u_matrix,tau_matrix).sum().item()
            return result

    def i2tc(self,i):
        '''
        传入i,返回t,c。 i是batch数
        '''
        t=math.floor(i/self.C)
        c=i%self.C
        return t,c

    def getCheckpoint(self,i):
        '''
        i表示第i次迭代，i in [0,TC-1] 若i==-1 表示的是初始化模型参数
        '''
        if self.mode=='Infer_k' or self.mode=='Infer_all':
            dirName=self.args.save_path[:-11]+'''TrainPhase/%s_%s_%s'''%(self.args.net_name,self.args.train_batch_size,self.args.lr)
            return torch.load(dirName+'/phase_%s.pth'%(i))
        elif self.mode=='SGD_dataPruning' or self.mode=='SGD_randomPruning':
            dirName=self.args.save_path.split('_')[0]+'''/%s_%s_%s'''%(self.args.net_name,self.args.train_batch_size,self.args.lr)
            return torch.load(dirName+'/phase_%s.pth'%(i))
        elif self.mode=='SGD_k':
            dirName='/home/yunxshi/Data/workspace/DataPruning/TrainPhase'+'''/%s_%s_%s'''%(self.args.net_name,self.args.train_batch_size,self.args.lr)
            return torch.load(dirName+'/phase_%s.pth'%(i))
        return torch.load(self.dirName+'/phase_%s.pth'%(i))

    def getLr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def getBatch_init(self):
        if not Trainer.getBatch_init_flag:
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            #存到内存
                Trainer.batch_dict[batch_idx]=(inputs, targets)
            Trainer.getBatch_init_flag=True
        else:
            pass

    def getBatch(self,i):
        '''
        i是第i个batch 
        '''
        t,c=self.i2tc(i)
        if self.use_getBatch_init:
            if not Trainer.getBatch_init_flag:
                self.getBatch_init()
            inputs, targets=Trainer.batch_dict[c]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            return inputs, targets
        else:
            if c in Trainer.batch_dict.keys():
                inputs, targets=Trainer.batch_dict[c]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                return inputs, targets
            else:
                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    if batch_idx==c:
                        Trainer.batch_dict[c]=(inputs, targets)
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        return inputs, targets
                    
    def saveParam(self,i,lr):
        if self.args.save_net:
            # 保存(epoch,batch_idx)之时的，网络中的参数，以及在验证集上的损失
            state = {'net': self.net.state_dict(),
                    'iter': i,
                    'lr':lr
                    }
            if not os.path.exists(self.dirName):
                os.makedirs(self.dirName)
            torch.save(state,self.dirName+'''/phase_%s.pth'''%(i))
        else:
            pass

    # Training
    def train(self,epoch):
        if self.mode=='SGD' or self.mode=='SGD_k':
            dataLoader=self.trainloader
        elif self.mode=='SGD_dataPruning':
            dataLoader=self.trainloader_dataPruning
        elif self.mode=='SGD_randomPruning':
            dataLoader=self.trainloader_randomPruning
        
        self.dataLoader_len=len(dataLoader)

        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataLoader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            #k!=-1 为counter-sgd，删掉Xk
            if self.mode=='SGD_k' and self.k!=-1 and batch_idx==self.q:
                inputs = inputs[torch.arange(inputs.size(0))!=self.index_inBatch]
                targets = targets[torch.arange(targets.size(0))!=self.index_inBatch]

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            loss=torch.mean(loss,dim =0)
            loss.backward()
            self.optimizer.step()


            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, self.dataLoader_len, 'TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1),100.*correct/total, correct, total))
            
            i=epoch*self.dataLoader_len+batch_idx
            
            self.saveParam(i,self.getLr())
            try:
                self.trainLoss_dict[i]=train_loss/(batch_idx+1)
                self.trainAcc_dict[i]=correct/total
            except:
                self.trainLoss_dict={}
                self.trainAcc_dict={}
                self.trainLoss_dict[i]=train_loss/(batch_idx+1)
                self.trainAcc_dict[i]= correct/total

            if self.test_everyIter:
                self.test(epoch,batch_idx)

    def summary(self):
        #输出每层网络参数信息
        summary(self.net,(3,32,32),batch_size=1,device=self.device)

    def test(self,epoch,batch_idx_fromTrain=0):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss=torch.mean(loss,dim =0)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if not self.test_everyIter:
            i=(epoch+1)*self.dataLoader_len-1
        elif self.test_everyIter:
            i=epoch*self.dataLoader_len+batch_idx_fromTrain
        try:
            self.testLoss_dict[i]=test_loss/(batch_idx+1)
            self.testAcc_dict[i]=correct/total
        except:
            self.testLoss_dict={}
            self.testAcc_dict={}
            self.testLoss_dict[i]=test_loss/(batch_idx+1)
            self.testAcc_dict[i]=correct/total

        return test_loss/(batch_idx+1),correct/total
        
            

    
    def inferAll(self,w_i=50):
        lossChangeAll=self.calLossChangeAll(w_i)
        return lossChangeAll

    def infer(self,w_i=50):
        self.prepareInfer(self.k,0)
        lossChange=self.calLossChange(w_i)
        return lossChange

    def train_phase(self):
        k=self.k
        #k为-1，是正常SGD
        if self.mode=='SGD':
            self.saveParam(-1,self.getLr())
        elif self.mode=='SGD_k':
            npyPath_test_loss=self.dirName+'/testLoss_SGD_%s.npy'%self.k
            if os.path.exists(npyPath_test_loss):
                self.testLoss_dict=np.load(npyPath_test_loss,allow_pickle=True).item()
                return
            else:
                #load初始参数
                checkpoint=self.getCheckpoint(-1)
                self.net.load_state_dict(checkpoint['net'])
        elif self.mode=='SGD_dataPruning':
            w_i=int(self.lossChangePath.split('.')[-2].split('_')[-1])
            npyPath_train_loss=self.dirName+'/trainLoss_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i)
            npyPath_train_acc=self.dirName+'/trainAcc_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i)
            npyPath_test_loss=self.dirName+'/testLoss_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i)
            npyPath_test_acc=self.dirName+'/testAcc_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i)
            if os.path.exists(npyPath_train_loss) and \
            os.path.exists(npyPath_train_acc) and \
            os.path.exists(npyPath_test_loss) and \
            os.path.exists(npyPath_test_acc):
                self.trainLoss_dict=np.load(npyPath_train_loss,allow_pickle=True).item()
                self.trainAcc_dict=np.load(npyPath_train_acc,allow_pickle=True).item()
                self.testLoss_dict=np.load(npyPath_test_loss,allow_pickle=True).item()
                self.testAcc_dict=np.load(npyPath_test_acc,allow_pickle=True).item()
                return 
            else:
                #load初始参数
                checkpoint=self.getCheckpoint(-1)
                self.net.load_state_dict(checkpoint['net'])
        elif self.mode=='SGD_randomPruning':
            npyPath_train_loss=self.dirName+'/trainLoss_%s.npy'%self.pruningSize
            npyPath_train_acc=self.dirName+'/trainAcc_%s.npy'%self.pruningSize
            npyPath_test_loss=self.dirName+'/testLoss_%s.npy'%self.pruningSize
            npyPath_test_acc=self.dirName+'/testAcc_%s.npy'%self.pruningSize

            if os.path.exists(npyPath_train_loss) and \
            os.path.exists(npyPath_train_acc) and \
            os.path.exists(npyPath_test_loss) and \
            os.path.exists(npyPath_test_acc):
                self.trainLoss_dict=np.load(npyPath_train_loss,allow_pickle=True).item()
                self.trainAcc_dict=np.load(npyPath_train_acc,allow_pickle=True).item()
                self.testLoss_dict=np.load(npyPath_test_loss,allow_pickle=True).item()
                self.testAcc_dict=np.load(npyPath_test_acc,allow_pickle=True).item()
                return
            else:
                #load初始参数
                checkpoint=self.getCheckpoint(-1)
                self.net.load_state_dict(checkpoint['net'])
        for epoch in range(self.args.epoch):
            self.train(epoch)
            if not self.test_everyIter:
                self.test(epoch)
            self.scheduler.step()        
        self.drawLoss()

    def realLossChange(self):
        '''
        keys从0到TC-1，分别计算迭代第0次~第TC-1次后的w的testloss，计算得到的losschange
        '''
        self.train_phase()
        realLossChange_dict={}
        SGD_testLoss_dict_npyPath=self.dirName.split('_')[0]+'''/%s_%s_%s'''%(self.args.net_name,self.args.train_batch_size,self.args.lr)+'/testLoss_SGD.npy'
        SGD_testLoss_dict=np.load(SGD_testLoss_dict_npyPath,allow_pickle=True).item()
        #/home/yunxshi/Data/workspace/DataPruning/TrainPhase/MyNet_1000_0.1
        for i,SGD_k_testLoss in self.testLoss_dict.items():
            SGD_loss=SGD_testLoss_dict[i]
            realLossChange_dict[i]=SGD_k_testLoss-SGD_loss
        return realLossChange_dict

    def drawLoss(self):
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)
        if self.mode=='SGD' :
            np.save(self.dirName+'/trainLoss_%s.npy'%'SGD',self.trainLoss_dict)
            np.save(self.dirName+'/trainAcc_%s.npy'%'SGD',self.trainAcc_dict)
            np.save(self.dirName+'/testLoss_%s.npy'%'SGD',self.testLoss_dict)
            np.save(self.dirName+'/testAcc_%s.npy'%'SGD',self.testAcc_dict)
            picPath=self.dirName+'/loss_acc.jpg'
        elif self.mode=='SGD_randomPruning':
            np.save(self.dirName+'/trainLoss_%s.npy'%(self.pruningSize),self.trainLoss_dict)
            np.save(self.dirName+'/trainAcc_%s.npy'%(self.pruningSize),self.trainAcc_dict)
            np.save(self.dirName+'/testLoss_%s.npy'%(self.pruningSize),self.testLoss_dict)
            np.save(self.dirName+'/testAcc_%s.npy'%(self.pruningSize),self.testAcc_dict)
            picPath=self.dirName+'/loss_acc_%s.jpg'%(self.pruningSize)
            
        elif self.mode=='SGD_dataPruning':
            w_i=int(self.lossChangePath.split('.')[-2].split('_')[-1])
            np.save(self.dirName+'/trainLoss_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i),self.trainLoss_dict)
            np.save(self.dirName+'/trainAcc_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i),self.trainAcc_dict)
            np.save(self.dirName+'/testLoss_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i),self.testLoss_dict)
            np.save(self.dirName+'/testAcc_%s_%s_%s.npy'%(self.pruningSize,self.pos,w_i),self.testAcc_dict)
            picPath=self.dirName+'/loss_acc_%s_%s_%s.jpg'%(self.pruningSize,self.pos,w_i)

        elif self.mode=='SGD_k':
            np.save(self.dirName+'/trainLoss_SGD_%s.npy'%self.k,self.trainLoss_dict)
            np.save(self.dirName+'/trainAcc_SGD_%s.npy'%self.k,self.trainAcc_dict)
            np.save(self.dirName+'/testLoss_SGD_%s.npy'%self.k,self.testLoss_dict)
            np.save(self.dirName+'/testAcc_SGD_%s.npy'%self.k,self.testAcc_dict)
            picPath=self.dirName+'/loss_acc_SGD_%s.jpg'%self.k

        plt.figure(1,figsize=(16,9))
        plt.subplot(2,1,1)
        plt.cla()
        plt.plot(self.trainLoss_dict.keys(), self.trainLoss_dict.values(), label="loss on train")
        plt.plot(self.testLoss_dict.keys(), self.testLoss_dict.values(), label="loss on valid")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.legend()

        plt.subplot(2,1,2)
        plt.cla()
        plt.plot(self.trainAcc_dict.keys(), self.trainAcc_dict.values(), label="acc on train")
        plt.plot(self.testAcc_dict.keys(), self.testAcc_dict.values(), label="acc on valid")
        plt.xlabel("iter")
        plt.ylabel("acc")
        plt.legend()

        plt.savefig(picPath)
if __name__ == "__main__":
    trainer = Trainer()
    trainer.randomPruning_train_phase()
    #trainer.dataPruning_train_phase()
    #trainer.train_phase()
    #trainer.infer()
