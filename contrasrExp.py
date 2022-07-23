import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--LossChange-path', default='''/workspace/DataPruning/LossChange/''', type=str, help='LossChange save path')
parser.add_argument('--p', default=0, type=int, help='Delete Xk in the p epoch')
parser.add_argument('--k', default=8888, type=int, help='Xk to delete')
parser.add_argument('--T', default=10, type=int, help='Epoch')
exp_args = parser.parse_args()

from SGD import *

class ContrastExp(object):
    def __init__(self):
        self.trainer = Trainer()
        self.args=exp_args

    def normal_train(self):
        self.trainer.train_phase()

    def get_loss_change_real(self,k):
        print('计算loss_change_real:')
        dir=self.args.LossChange_path+'''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)
        if not os.path.exists(dir):
            os.makedirs(dir)
        npy_path=dir+'/realLossChange_%s.npy'%k
        if os.path.exists(npy_path):
            return np.load(npy_path,allow_pickle=True).item()
        else:
            realLossChange=SGD_k(k)
            np.save(npy_path,realLossChange)
            return realLossChange

    def get_LossChange_apx(self,k,begin_w,end_w,use_apx_ju=False):
        '''
        w的范围是[1,TC]
        '''
        print('计算loss_change_apx:')
        dir=self.args.LossChange_path+'''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if use_apx_ju:
            npy_path=dir+'/apxLossChange_%s_%s_%s_apxJu.npy'%(k,begin_w,end_w)
        else:
            npy_path=dir+'/apxLossChange_%s_%s_%s.npy'%(k,begin_w,end_w)
        if os.path.exists(npy_path):
            return np.load(npy_path,allow_pickle=True).item()
        else:
            apxLossChange={}
            #这里的i表示的是迭代次数,i的范围是[0,TC-1]，所计算出来的是[0,TC]
            for i in range(begin_w,end_w+1):
                apxLossChange[i]=infer_k(k,i,use_apx_ju)
            np.save(npy_path,apxLossChange)
            return apxLossChange

    def calLossChangeAll(self,w_i=30,kList=[]):
        print('计算loss_change_apx_all:')
        dir=self.args.LossChange_path+'''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)
        if not os.path.exists(dir):
            os.makedirs(dir)

        npy_path=dir+'/apxLossChangeAll_%s.npy'%w_i
        if os.path.exists(npy_path):
            return np.load(npy_path,allow_pickle=True)
        else:
            lossChange_dict=infer_all(w_i,kList)
            np.save(npy_path,lossChange_dict)
            return lossChange_dict


    def experiment1(self,k,begin_w,end_w):
        #begin_w 和 end_w 表示的是w的范围，应该[1,TC]之间，对应real应该是[0,TC-1]
        dir=self.args.LossChange_path+'''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)
        jpgPath=dir+'/lossChange_%s_exp1.jpg'%(k)
        
        begin_i=begin_w-1
        end_i=end_w-1
        loss_change_apx=self.get_LossChange_apx(k,begin_w,end_w)
        loss_change_real=self.get_loss_change_real(k)
        
        
        plt.figure(1,figsize=(16,9))
        plt.cla()
        plt.plot(range(len(loss_change_real)), loss_change_real.values(), label="loss change real")
        plt.plot(range(begin_i,end_i+1), loss_change_apx.values(), label="loss change apx")

        print('len(loss_change_real)',len(loss_change_real))
        for i in range(9,len(loss_change_real),10):
            losschange_value=list(loss_change_real.values())[i]
            if losschange_value<=0:
                c='orange'
            else:
                c='purple'
            plt.scatter([i], [losschange_value], s=25, c=c)  # stroke, colour

        plt.axhline(y=0, xmin=0, xmax=1,c='green')

        plt.xlabel("iter")
        plt.ylabel("loss change")
        plt.legend()
        plt.savefig(jpgPath)

    def experiment3(self,k,begin_w,end_w):
        #begin_w 和 end_w 表示的是w的范围，w的范围是[0,TC]，在apx中我们应该[1,TC]之间都可以，对应real应该是[0,TC-1]
        dir=self.args.LossChange_path+'''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)
        jpgPath=dir+'/lossChange_%s_exp3.jpg'%(k)
        
        begin_i=begin_w-1
        end_i=end_w-1
        loss_change_apx=self.get_LossChange_apx(k,begin_w,end_w,use_apx_ju=True)
        loss_change_real=self.get_loss_change_real(k)
        print('len(loss_change_real)',len(loss_change_real))
        print(loss_change_real)
        print('len(loss_change_apx)',len(loss_change_apx))
        print(loss_change_apx)

        plt.figure(1,figsize=(16,9))
        plt.plot(range(len(loss_change_real)), loss_change_real.values(), label="loss change real")
        plt.plot(range(begin_i,end_i+1), loss_change_apx.values(), label="loss change apx")
        plt.xlabel("iter")
        plt.ylabel("loss change")
        plt.legend()
        plt.savefig(jpgPath)

    def experiment(self,k):
        begin_w=1
        end_w=30
        iter_range=range(begin,end+1)
        loss_change_apx=[self.trainer.calLossChangeAll(i=j,kList=[k])[k] for j in iter_range]
        loss_change_real=list(self.get_loss_change_real(k).values())
        print(loss_change_real)
        print(loss_change_apx)
        plt.figure(1,figsize=(16,9))
        plt.plot(iter_range, loss_change_real[begin:end+1], label="loss_change_real")
        plt.plot(iter_range, loss_change_apx, label="loss_change_apx")
        plt.xlabel("iter")
        plt.ylabel("loss change")
        plt.legend()
        plt.savefig(self.args.LossChange_path+'losschange_%s.jpg'%k)
    
    def experiment4(self,pruningSize=0.6,pos=1,w_i=100):
        dirName='''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)

        loss_dataPruning,acc_dataPruning=SGD_dataPruning(pruningSize,pos,w_i)
        loss_randomPruning,acc_randomPruning=SGD_randomPruning(pruningSize)
        loss_SGD,acc_SGD=get_SGD_loss_acc()

        plt.clf()
        plt.figure(1,figsize=(16,9))

        plt.plot(range(len(loss_dataPruning)), loss_dataPruning.values(), label="loss_dataPruning")
        plt.plot(range(len(loss_randomPruning)), loss_randomPruning.values(), label="loss_randomPruning")
        plt.plot(range(len(loss_SGD)), loss_SGD.values(), label="loss_SGD")

        if list(acc_dataPruning.values())[0]//10!=0:
            plt.plot(range(len(acc_dataPruning)), np.array(list(acc_dataPruning.values()))/100, label="acc_dataPruning")
        else:
            plt.plot(range(len(acc_dataPruning)), acc_dataPruning.values(), label="acc_dataPruning")
        if list(acc_randomPruning.values())[0]//10!=0:
            plt.plot(range(len(acc_randomPruning)),  np.array(list(acc_randomPruning.values()))/100, label="acc_randomPruning")
        else:
            plt.plot(range(len(acc_randomPruning)),  acc_randomPruning.values(), label="acc_randomPruning")
        if list(acc_SGD.values())[1]//10!=0:
             plt.plot(range(len(acc_SGD)), np.array(list(acc_SGD.values()))/100, label="acc_SGD")
        else:
            plt.plot(range(len(acc_SGD)), acc_SGD.values(), label="acc_SGD")

        plt.xlabel("iter")
        plt.ylabel("loss and acc")
        plt.legend()

        LossChange_path=self.args.LossChange_path+dirName
        if not os.path.exists(LossChange_path):
            os.makedirs(LossChange_path)
        plt.savefig(LossChange_path+'/loss_acc_compare_%s_%s_%s.jpg'%(pruningSize,pos,w_i))

def ana_all():
    fName='apxLossChangeAll_105'
    dirName='/workspace/DataPruning/LossChange/MyNet_1000_0.1/'
    filePath=dirName+fName+'.npy'
    lossChange_dict=np.load(filePath, allow_pickle=True).item()
    lossChange_dict_sorted=sorted(lossChange_dict.items(), key = lambda kv:(kv[1], kv[0]))
    #画图
    df= pd.DataFrame(lossChange_dict_sorted, columns=['iter','lossChange'])
    fig = df['lossChange'].hist()
    fig.figure.savefig(dirName+fName+'.jpg') # 保存

def tmp():
    filePath='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_100_1.npy'
    lossChange_dict1=np.load(filePath, allow_pickle=True).item()

    filePath='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_100_2.npy'
    lossChange_dict2=np.load(filePath, allow_pickle=True).item()

    lossChange_dict1.update(lossChange_dict2)

    np.save('/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_100.npy',lossChange_dict1)

def tmp2():
    filePath='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_100.npy'
    lossChange_dict=np.load(filePath, allow_pickle=True).item()
    print(lossChange_dict[666])
    print(lossChange_dict[777])
    print(lossChange_dict[5555])
    print(lossChange_dict[6666])
    print(lossChange_dict[7777])
    print(lossChange_dict[8888])

def tmp3():
    filePath4='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_400.npy'
    filePath5='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_500.npy'
    filePath6='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_600.npy'
    filePath='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_400500600.npy'
    lossChange_dict4=np.load(filePath4, allow_pickle=True).item()
    lossChange_dict5=np.load(filePath5, allow_pickle=True).item()
    lossChange_dict6=np.load(filePath6, allow_pickle=True).item()

    lossChange_dict={}
    for key in lossChange_dict4.keys():
        value4=lossChange_dict4[key]
        value5=lossChange_dict5[key]
        value6=lossChange_dict6[key]
        value=0.2*value4+0.3*value5+0.5*value6
        lossChange_dict[key]=value
    np.save(filePath,lossChange_dict)



if __name__ == "__main__":
    #tmp3()
    ConExp = ContrastExp()
    # kList1=list(range(1,5000+1))
    # kList2=list(range(5001,10000+1))
    #ConExp.calLossChangeAll(100,kList1)
    #ConExp.get_loss_change_real(k=exp_args.k)
    #ConExp.experiment3(k=8888,begin_w=50,end_w=55)
    # ConExp.calLossChangeAll(i=30)

    # ConExp.experiment1(6236,80,100)
    # ConExp.experiment1(233,80,100)
    # ConExp.experiment1(110,80,100)
    # ConExp.experiment1(119,80,100)
    # ConExp.experiment1(777,80,100)
    #ConExp.experiment3(7777,100,120)
    # ConExp.experiment1(5555,80,100)
    # ConExp.experiment1(6666,80,100)
    # ConExp.experiment1(666,80,100)

    #trainer.infer()
    # ConExp.experiment4(pruningSize=0.8,pos=2,w_i=400)
    # ConExp.experiment4(pruningSize=0.8,pos=3,w_i=400)

    # ConExp.experiment4(pruningSize=0.05,pos=3,w_i=500)



    #ConExp.experiment4(pruningSize=0.1,pos=1,w_i=400500600)
    ConExp.experiment4(pruningSize=0.05,pos=1,w_i=400500600)
    ConExp.experiment4(pruningSize=0.2,pos=1,w_i=400500600)
    ConExp.experiment4(pruningSize=0.3,pos=1,w_i=400500600)
    # ConExp.experiment4(pruningSize=0.2,pos=3,w_i=500)
    # ConExp.experiment4(pruningSize=0.3,pos=3,w_i=500)
    # ConExp.experiment4(pruningSize=0.4,pos=3,w_i=500)
    # ConExp.experiment4(pruningSize=0.5,pos=3,w_i=500)
    # ConExp.experiment4(pruningSize=0.6,pos=3,w_i=500)
    # ConExp.experiment4(pruningSize=0.7,pos=3,w_i=500)
    # ConExp.experiment4(pruningSize=0.8,pos=3,w_i=500)

    # ConExp.experiment4(pruningSize=0.1,pos=1,w_i=400)
    # ConExp.experiment4(pruningSize=0.2,pos=1,w_i=400)
    # ConExp.experiment4(pruningSize=0.3,pos=1)
    # ConExp.experiment4(pruningSize=0.5,pos=1)
    # ConExp.experiment4(pruningSize=0.6,pos=1)
    # ConExp.experiment4(pruningSize=0.7,pos=1)

    # ConExp.experiment4(pruningSize=0.05,pos=2)
    # ConExp.experiment4(pruningSize=0.05,pos=3)
    #ConExp.experiment4(pruningSize=0.1,pos=1)
    # ConExp.experiment4(pruningSize=0.1,pos=2)
    # ConExp.experiment4(pruningSize=0.1,pos=3)
    #ConExp.experiment4(pruningSize=0.2,pos=1)
    # ConExp.experiment4(pruningSize=0.2,pos=2)
    # ConExp.experiment4(pruningSize=0.2,pos=3)
    #ConExp.experiment4(pruningSize=0.4,pos=1)
    # ConExp.experiment4(pruningSize=0.4,pos=2)
    # ConExp.experiment4(pruningSize=0.4,pos=3)
    #ConExp.experiment4(pruningSize=0.5,pos=1)
    # ConExp.experiment4(pruningSize=0.5,pos=2)
    # ConExp.experiment4(pruningSize=0.5,pos=3)
    # ConExp.experiment4(pruningSize=0.6,pos=1)
    # ConExp.experiment4(pruningSize=0.6,pos=2)
    # ConExp.experiment4(pruningSize=0.6,pos=3)
    #ConExp.experiment4(pruningSize=0.8,pos=1)
    # ConExp.experiment4(pruningSize=0.05,pos=1)
    #ConExp.experiment4(pruningSize=0.1,pos=1)
    # ConExp.experiment4(pruningSize=0.2,pos=1)
    # ConExp.experiment4(pruningSize=0.3,pos=1)
    # ConExp.experiment4(pruningSize=0.4,pos=1)
    # # ConExp.experiment4(pruningSize=0.5,pos=1)
    # ConExp.experiment4(pruningSize=0.8,pos=1)
    # ConExp.experiment4(pruningSize=0.05,pos=2)
    # ConExp.experiment4(pruningSize=0.1,pos=2)
    # ConExp.experiment4(pruningSize=0.2,pos=2)
    # ConExp.experiment4(pruningSize=0.3,pos=2)
    # ConExp.experiment4(pruningSize=0.4,pos=2)
    # ConExp.experiment4(pruningSize=0.5,pos=2)
    # ConExp.experiment4(pruningSize=0.8,pos=2)
    # ConExp.experiment4(pruningSize=0.05,pos=3)  
    #ConExp.experiment4(pruningSize=0.1,pos=1)
    # ConExp.experiment4(pruningSize=0.2,pos=3)
    # ConExp.experiment4(pruningSize=0.3,pos=3)
    #ConExp.experiment4(pruningSize=0.4,pos=1)
    # ConExp.experiment4(pruningSize=0.5,pos=3)
    #ConExp.experiment4(pruningSize=0.8,pos=1)