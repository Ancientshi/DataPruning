from main import *

parser = argparse.ArgumentParser(description='Task args')
parser.add_argument('--save-net', default=1, type=int, help='Save net')
parser.add_argument('--pos', default=1, type=int, help='DataPruning pos')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train-batch-size', default=128, type=int, help='train batch size')
parser.add_argument('--mode', default='SGD', type=str, help='Task option')
parser.add_argument('--net-name', default='ResNet50', type=str, help='used net name')
task_args = parser.parse_args()

def SGD(save=1):
    task_args.save_net=save
    trainer = Trainer(task_args=task_args)
    trainer.train_phase()

def SGD_k(k,save=0):
    task_args.save_net=save
    task_args.mode='SGD_k'
    task_args.k=k
    trainer = Trainer(task_args=task_args)
    #0: 4.172325134277344e-07的形式
    realLossChange_dict=trainer.realLossChange()
    return realLossChange_dict

def infer_k(k,w_i,use_apx_ju=False):
    '''
    计算删去k，在W为w_i，近似计算出来的loss change
    '''
    task_args.mode='Infer_k'
    task_args.k=k
    trainer = Trainer(task_args=task_args)
    trainer.use_apx_ju=use_apx_ju
    loss_change_apx_wi=trainer.infer(w_i=w_i)
    print('loss_change_apx_w%s'%w_i,loss_change_apx_wi)
    return loss_change_apx_wi

def infer_all(w_i,kList=[]):
    '''
    计算在W为w_i，近似计算删去kList中的k造成的loss change
    '''
    task_args.mode='Infer_all'
    trainer = Trainer(task_args=task_args)
    trainer.kList=kList
    loss_change_apx_wi=trainer.inferAll(w_i=w_i)
    return loss_change_apx_wi

def SGD_dataPruning(pruningSize=0.6,pos=1,w_i=100):
    lossChangePath='/workspace/DataPruning/LossChange/MyNet_1000_0.1/apxLossChangeAll_%s.npy'%w_i
    task_args.mode='SGD_dataPruning'
    task_args.lossChangePath=lossChangePath
    task_args.pruningSize=pruningSize
    task_args.pos=pos
    trainer = Trainer(task_args=task_args)
    return trainer.train_phase()

def SGD_randomPruning(pruningSize=0.6):
    task_args.mode='SGD_randomPruning'
    task_args.pruningSize=pruningSize
    trainer = Trainer(task_args=task_args)
    return trainer.train_phase()

def get_SGD_loss_acc():
    dirName='''%s_%s_%s'''%(task_args.net_name,task_args.train_batch_size,task_args.lr)
    npyPath_loss='''/workspace/DataPruning/TrainPhase/%s/testLoss_SGD.npy'''%dirName
    npyPath_acc='''/workspace/DataPruning/TrainPhase/%s/testAcc_SGD.npy'''%dirName
    return np.load(npyPath_loss,allow_pickle=True).item(),np.load(npyPath_acc,allow_pickle=True).item()

if __name__ == "__main__":
    #SGD(0)
    SGD_dataPruning(0.5,1,400500600)
    
    # SGD_dataPruning()
    # SGD_randomPruning()
    # infer_k(1,1)
    # infer_k(1,2)
    # infer_k(1,3)
    # infer_k(1,7)

    #{38: -3.743982e-05, 39: -2.2695916e-05, 40: -0.00025426497, 41: -9.715123e-05, 42: -8.422554e-05}
    # [{8888: -4.010486e-05}, {8888: -2.4470704e-05}, {8888: -0.00025107007}, {8888: -9.492615e-05}, {8888: -8.432305e-05}]


    # 37: -3.1828880310058594e-05, 38: -0.0001747608184814453, 39: -0.00027561187744140625, 40: -8.70823860168457e-05, 41: -7.736682891845703e-05
    '''
    计算当v=0时的式子, prod=tensor(-1.5260e-05, device='cuda:0')
    计算当v=1时的式子, prod=tensor(-1.2353e-05, device='cuda:0')
    计算当v=2时的式子, prod=tensor(-9.8868e-06, device='cuda:0')
    loss_change_apx_w38 -3.7499925e-05
    '''
    #infer_k(8888,38)


    '''
    u_init tensor([[-4.9002e-04, -5.3311e-04, -7.7896e-04,  ...,  2.8838e-05,
          1.5921e-02, -1.5921e-02]], device='cuda:0')

    {2: tensor([[-6.8637e-05, -1.9970e-04, -5.9445e-04,  ...,  1.9941e-05,
          4.2239e-03, -4.2239e-03]], device='cuda:0'), 
    1: tensor([[ 2.0051e-04, -2.6462e-05, -6.4447e-04,  ..., -5.8690e-05,
          9.1327e-03, -9.1327e-03]], device='cuda:0'), 
    0: tensor([[ 7.4374e-04,  3.7047e-04, -5.9933e-04,  ..., -9.3837e-05,
          1.9095e-02, -1.9095e-02]], device='cuda:0')}

    计算当v=0时的式子, prod=tensor(-1.5268e-05, device='cuda:0')
    计算当v=1时的式子, prod=tensor(-1.2325e-05, device='cuda:0')
    计算当v=2时的式子, prod=tensor(-9.8469e-06, device='cuda:0')
    loss_change_apx_w38 -3.743982e-05
    '''
    #infer_k(8888,38,True)



    #计算当v=0时的式子, prod=tensor(-1.7933e-05, device='cuda:0')
    # 计算当v=1时的式子, prod=tensor(-1.2325e-05, device='cuda:0')
    # 计算当v=2时的式子, prod=tensor(-9.8469e-06, device='cuda:0')
    #loss_change_apx_w38 {8888: -4.010486e-05}
    '''
    u_init tensor([[-4.9002e-04, -5.3311e-04, -7.7896e-04,  ...,  2.8838e-05,
          1.5921e-02, -1.5921e-02]], device='cuda:0')

    {'0_8': tensor([[ 0.0008,  0.0005, -0.0006,  ..., -0.0003,  0.0198, -0.0198]],
    device='cuda:0'), 
    '2_8': tensor([[-6.8637e-05, -1.9970e-04, -5.9445e-04,  ...,  1.9941e-05,
    4.2239e-03, -4.2239e-03]], device='cuda:0'), 
    '1_8': tensor([[ 2.0051e-04, -2.6462e-05, -6.4447e-04,  ..., -5.8690e-05,
    9.1327e-03, -9.1327e-03]], device='cuda:0')}
    '''

    '''
正在计算u*Z^10
j=10
u: tensor([[ 7.4374e-04,  3.7047e-04, -5.9933e-04,  ..., -9.3837e-05,
          1.9095e-02, -1.9095e-02]], device='cuda:0')

正在计算u*Z^9
j=9
u: tensor([[ 0.0008,  0.0005, -0.0006,  ..., -0.0003,  0.0198, -0.0198]],
       device='cuda:0')
    '''
    #print([ infer_all(k,[8888]) for k in range(38,43)])