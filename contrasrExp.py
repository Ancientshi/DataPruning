import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dataloader import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--LossChange-path', default='''/home/yunxshi/Data/workspace/DataPruning/LossChange/''', type=str, help='LossChange save path')
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
            realLossChange=SGD_k(k,0,True)
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
        jpgPath=dir+'/lossChange_%s_%s_%s_exp1.jpg'%(k,begin_w,end_w)
        
        begin_i=begin_w-1
        end_i=end_w-1
        loss_change_apx=self.get_LossChange_apx(k,begin_w,end_w)
        loss_change_real=self.get_loss_change_real(k)
        
        
        plt.figure(1,figsize=(16,9))
        plt.cla()
        plt.plot(loss_change_real.keys(), loss_change_real.values(), label="loss change real")
        plt.plot(np.array(list(loss_change_apx.keys()))-1, loss_change_apx.values(), label="loss change apx")

        loss_change_apx_keys=list(loss_change_apx.keys())
        loss_change_apx_values=list(loss_change_apx.values())
        for apx_key in loss_change_apx_keys:
            if apx_key%(50000//task_args.train_batch_size)==0:
                real_key=apx_key-1
                lossChange_real=loss_change_real[real_key]
                lossChange_apx=loss_change_apx[apx_key]
                plt.scatter([real_key], [lossChange_real], s=25, c='orange',label="loss change real")  # stroke, colour
                plt.scatter([real_key], [lossChange_apx], s=25, c='purple',label="loss change apx")  # stroke, colour

        plt.axhline(y=0, xmin=0, xmax=1,c='green')

        plt.xlabel("iter")
        plt.ylabel("loss change")
        plt.legend()
        plt.savefig(jpgPath)

    def experiment3(self,k,begin_w,end_w,use_apx_ju=True):
        #begin_w 和 end_w 表示的是w的范围，w的范围是[0,TC]，w_0的losschange是0（w_0是初始化参数），
        #在这里中我们具有实际意义的w范围应该[1,TC]之间，loss_change_apx的keys就是这个区间，
        #而loss_change_real的区间是[0,TC-1]，表示的是第0~TC-1次迭代
        dir=self.args.LossChange_path+'''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)
        jpgPath=dir+'/lossChange_%s_%s_%s_exp3.jpg'%(k,begin_w,end_w)
        
        begin_i=begin_w-1
        end_i=end_w-1
        loss_change_apx=self.get_LossChange_apx(k,begin_w,end_w,use_apx_ju)
        loss_change_real=self.get_loss_change_real(k)
        print('len(loss_change_real)',len(loss_change_real))
        print(loss_change_real)
        print('len(loss_change_apx)',len(loss_change_apx))
        print(loss_change_apx)

        plt.figure(1,figsize=(16,9))
        plt.cla()
        # plt.plot(list(loss_change_real.keys())[begin_i:end_i+1], list(loss_change_real.values())[begin_i:end_i+1], label="loss change real")
        plt.plot(list(loss_change_real.keys()), list(loss_change_real.values()), label="loss change real")
        plt.plot(np.array(list(loss_change_apx.keys()))-1, loss_change_apx.values(), label="loss change apx")

        loss_change_apx_keys=list(loss_change_apx.keys())
        loss_change_apx_values=list(loss_change_apx.values())
        for apx_key in loss_change_apx_keys:
            if apx_key%(50000//task_args.train_batch_size)==0:
                real_key=apx_key-1
                print('real_key',real_key)
                lossChange_real=loss_change_real[real_key]
                lossChange_apx=loss_change_apx[apx_key]
                plt.scatter([real_key], [lossChange_real], s=25, c='orange',label="loss change real")  # stroke, colour
                plt.scatter([real_key], [lossChange_apx], s=25, c='purple',label="loss change apx")  # stroke, colour

        plt.axhline(y=0, xmin=0, xmax=1,c='green')


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

        train_loss_dataPruning,train_acc_dataPruning,test_loss_dataPruning,test_acc_dataPruning=SGD_dataPruning(pruningSize,pos,w_i)
        train_loss_randomPruning,train_acc_randomPruning,test_loss_randomPruning,test_acc_randomPruning=SGD_randomPruning(pruningSize)
        train_loss_SGD,train_acc_SGD,test_loss_SGD,test_acc_SGD=get_SGD_loss_acc()

        
        plt.figure(1,figsize=(16,10))


        plt.subplot(2,2,1)
        plt.cla()
        plt.plot(train_loss_dataPruning.keys(), train_loss_dataPruning.values(), label="train_loss_dataPruning")
        plt.plot(train_loss_randomPruning.keys(), train_loss_randomPruning.values(), label="train_loss_randomPruning")
        plt.plot(train_loss_SGD.keys(), train_loss_SGD.values(), label="train_loss_SGD")
        plt.xlabel("iter")
        plt.ylabel("train loss")
        plt.legend()

        plt.subplot(2,2,2)
        plt.cla()
        plt.plot(test_loss_dataPruning.keys(), test_loss_dataPruning.values(), label="test_loss_dataPruning")
        plt.plot(test_loss_randomPruning.keys(), test_loss_randomPruning.values(), label="test_loss_randomPruning")
        plt.plot(test_loss_SGD.keys(), test_loss_SGD.values(), label="test_loss_SGD")
        plt.xlabel("iter")
        plt.ylabel("test loss")
        plt.legend()

        plt.subplot(2,2,3)
        plt.cla()
        plt.plot(train_acc_dataPruning.keys(), train_acc_dataPruning.values(), label="train_acc_dataPruning")
        plt.plot(train_acc_randomPruning.keys(), train_acc_randomPruning.values(), label="train_acc_randomPruning")
        plt.plot(train_acc_SGD.keys(), train_acc_SGD.values(), label="train_acc_SGD")
        plt.xlabel("iter")
        plt.ylabel("train acc")
        plt.legend()

        plt.subplot(2,2,4)
        plt.cla()
        plt.plot(test_acc_dataPruning.keys(), test_acc_dataPruning.values(), label="test_acc_dataPruning")
        plt.plot(test_acc_randomPruning.keys(), test_acc_randomPruning.values(), label="test_acc_randomPruning")
        plt.plot(test_acc_SGD.keys(), test_acc_SGD.values(), label="test_acc_SGD")
        plt.xlabel("iter")
        plt.ylabel("test acc")
        plt.legend()

        LossChange_path=self.args.LossChange_path+dirName
        if not os.path.exists(LossChange_path):
            os.makedirs(LossChange_path)
        plt.savefig(LossChange_path+'/loss_acc_compare_%s_%s_%s.jpg'%(pruningSize,pos,w_i))

def ana_all(w_i):
    fName='apxLossChangeAll_%s'%w_i
    dirName='LossChange/MyNet_1000_0.1/'
    filePath=dirName+fName+'.npy'
    lossChange_dict=np.load(filePath, allow_pickle=True).item()
    print(list(lossChange_dict.items())[:5])
    lossChange_dict_sorted=sorted(lossChange_dict.items(), key = lambda kv:(kv[1], kv[0]))
    #画图
    df= pd.DataFrame(lossChange_dict_sorted, columns=['iter','lossChange'])
    fig = df['lossChange'].hist()
    fig.figure.savefig(dirName+fName+'distribtion.jpg') # 保存
    #-----
    cifar10_dataset=CIFAR10_DataPruning(root='/home/yunxshi/Data/cifar-10-python',
            train= True,
            download= False,
            datapruning=False)
    class_array=np.array(cifar10_dataset.targets).reshape(-1,1)
    plt.figure(figsize=(16,8))
    plt.cla()
    plt.scatter( list(lossChange_dict.values()), np.zeros(50000) , c=class_array)
    plt.savefig(dirName+fName+'scatter.jpg')



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
    filePath4='LossChange/MyNet_1000_0.1/apxLossChangeAll_400.npy'
    filePath5='LossChange/MyNet_1000_0.1/apxLossChangeAll_500.npy'
    filePath6='LossChange/MyNet_1000_0.1/apxLossChangeAll_600.npy'
    filePath='LossChange/MyNet_1000_0.1/apxLossChangeAll_400500600.npy'
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

def elbow():
    dir='MyNet_1000_0.1'
    w_i=900
    npyFilePath='LossChange/%s/apxLossChangeAll_%s.npy'%(dir,w_i)
    elbowFilePath='LossChange/%s/elbow_%s.jpg'%(dir,w_i)
    


    lossChange_dict=np.load(npyFilePath, allow_pickle=True).item()

    cifar10_dataset=CIFAR10_DataPruning(root='/home/yunxshi/Data/cifar-10-python',
            train= True,
            download= False,
            datapruning=False)
    
    class_array=np.array(cifar10_dataset.targets).reshape(-1,1)
    lossChange_array = np.array(list(lossChange_dict.values())).reshape(-1,1)

    tmp_array=np.zeros(50000).reshape(-1,1)
    data=np.concatenate((tmp_array,lossChange_array),1)
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16,8))
    plt.cla()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(elbowFilePath)

def cluser():
    dir='MyNet_1000_0.1'
    w_i=900
    npyFilePath='LossChange/%s/apxLossChangeAll_%s.npy'%(dir,w_i)
    kmeansPath='LossChange/%s/kmeans_%s.jpg'%(dir,w_i)

    lossChange_dict=np.load(npyFilePath, allow_pickle=True).item()
    lossChange_array = np.array(list(lossChange_dict.values())).reshape(-1,1)
    cifar10_dataset=CIFAR10_DataPruning(root='/home/yunxshi/Data/cifar-10-python',
            train= True,
            download= False,
            datapruning=False)
    class_array=np.array(cifar10_dataset.targets).reshape(-1,1)
    tmp_array=np.zeros(50000).reshape(-1,1)
    data=np.concatenate((tmp_array,lossChange_array),1)

    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(data)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    kmeansLabel_array=kmeanModel.predict(data)

    plt.figure(1,figsize=(16,10))
    plt.scatter(class_array, lossChange_array, c=kmeansLabel_array)
    plt.xlabel('calss')
    plt.ylabel('loss change')
    plt.legend()
    plt.title('Kmeans result')
    plt.savefig(kmeansPath)

    # kmeans.labels_ 
    # kmeans.predict([[0, 0], [12, 3]])

    # kmeans.cluster_centers_



if __name__ == "__main__":
    ConExp = ContrastExp()
    # ConExp.experiment4(pruningSize=0.6,pos=1,w_i=900)
    # ConExp.experiment4(pruningSize=0.7,pos=1,w_i=900)
    # ConExp.experiment4(pruningSize=0.8,pos=1,w_i=900)
    
    #ConExp.calLossChangeAll(10)
    #ana_all(10)
    ConExp.experiment3(8888,1,50)
    #ConExp.experiment3(1,51,100)
    #ConExp.experiment3(1,101,150)
    