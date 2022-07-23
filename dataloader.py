import numpy as np
from typing import Optional, Callable, Any, Tuple
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as transforms
import torch
import random
from PIL import Image
import os
random.seed(0)
np.random.seed(0)

# cifar10_train_path='''/workspace/DataPruning/cifar-10-python/train'''
# cifar10_test_path='''/workspace/DataPruning/cifar-10-python/test'''

# transform= transforms.Compose([
#     #transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

# with open(cifar10_train_path+'/data_batch_1', 'rb') as fo:
#     dict = pickle.load(fo, encoding='bytes')

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


# def GetPhoto(pixel):
#     assert len(pixel) == 3072
#     r = pixel[0:1024]; r = np.reshape(r, [1,32, 32])
#     g = pixel[1024:2048]; g = np.reshape(g, [1,32, 32])
#     b = pixel[2048:3072]; b = np.reshape(b, [1,32, 32])
#     photo = np.concatenate([r, g, b], 0)
#     # photo=photo.transpose(1,2,0)
#     # #(3, 32, 32)
#     # photo = Image.fromarray(photo) 
#     # photo=transform(photo)
#     photo=torch.tensor(photo, dtype=torch.float32)/255
#     return photo


# def getTrainDataByKeyword(keyword, size=(224, 224), normalized=False, filelist=[],index=-1):
#     '''
#     :param keyword:'data', 'labels', 'batch_label', 'filenames', 表示需要返回的项目
#     :param size:当keyword 是data时，表示需要返回的图片的尺寸
#     :param normalized:当keyword是data时，表示是否需要归一化
#     :param filelist:一个list， 表示需要使用的文件对象，仅1， 2， 3， 4， 5是有效的，其他数字无效
#     :return:需要返回的数据对象。'data'表示需要返回像素数据。'labels'表示需要返回标签数据。'batch_label'表示需要返回文件标签数据。'filenames'表示需要返回文件的文件名信息。
#     '''

#     keyword = str.encode(keyword)

#     assert keyword in [b'data', b'labels', b'batch_label', b'filenames']
#     assert type(filelist) is list and len(filelist) != 0
#     assert type(normalized) is bool
#     assert type(size) is tuple

#     files = []
#     for i in filelist:
#         if 1 <= i <= 5 and i not in files:
#             files.append(i)

#     if len(files) == 0:
#         raise ValueError("No valid input files!")

#     if keyword == b'data':
#         data = []
#         for i in files:
#             data.append(unpickle(cifar10_train_path+"/data_batch_%d" % i)[b'data'])
#         data = np.concatenate(data, 0)
#         return data
#         pass
#     if keyword == b'labels':
#         labels = []
#         for i in files:
#             labels += unpickle(cifar10_train_path+"/data_batch_%d" % i)[b'labels']
#         return labels
#         pass
#     elif keyword == b'batch_label':
#         batch_label = []
#         for i in files:
#             batch_label.append(unpickle(cifar10_train_path+"/data_batch_%d" % i)[b'batch_label'])
#         return batch_label
#         pass
#     elif keyword == b'filenames':
#         filenames = []
#         for i in files:
#             filenames += unpickle(cifar10_train_path+"/data_batch_%d" % i)[b'filenames']
#         return filenames
#         pass
#     pass


# def getTestDataByKeyword(keyword, size=(224, 224), normalized=False):
#     '''
#     :param keyword:'data', 'labels', 'batch_label', 'filenames', 表示需要返回的项目
#     :param size:当keyword 是data时，表示需要返回的图片的尺寸
#     :param normalized:当keyword是data时，表示是否需要归一化
#     :return:需要返回的数据对象。'data'表示需要返回像素数据。'labels'表示需要返回标签数据。'batch_label'表示需要返回文件标签数据。'filenames'表示需要返回文件的文件名信息。
#     '''
#     keyword = str.encode(keyword)

#     assert keyword in [b'data', b'labels', b'batch_label', b'filenames']
#     assert type(size) is tuple
#     assert type(normalized) is bool

#     batch_label = []
#     filenames = []

#     batch_label.append(unpickle(cifar10_test_path+"/test_batch")[b'batch_label'])
#     labels = unpickle(cifar10_test_path+"/test_batch")[b'labels']
#     data = unpickle(cifar10_test_path+"/test_batch")[b'data']
#     filenames += unpickle(cifar10_test_path+"/test_batch")[b'filenames']

#     label = keyword
#     if label == b'data':
#         return data
#         pass
#     elif label == b'labels':
#         return labels
#         pass
#     elif label == b'batch_label':
#         return batch_label
#         pass
#     elif label == b'filenames':
#         return filenames
#         pass
#     else:
#         raise NameError
#     pass


# class trainset(Dataset):
#     def __init__(self,positive='automobile',negative='truck', size=(224, 224)):
#         self.class_lable_dict={
#             'airplane':0,
#             'automobile':1,
#             'bird':2,
#             'cat':3,
#             'deer':4,
#             'dog':5,
#             'frog':6,
#             'horse':7,
#             'ship':8,
#             'truck':9
#         }
#         self.positive_lable=self.class_lable_dict[positive]
#         self.negative_lable=self.class_lable_dict[negative]
#         self.size=size

#         labels=getTrainDataByKeyword(keyword='labels', size=self.size, normalized=False, filelist=[1,2,3,4,5])

#         images=getTrainDataByKeyword(keyword='data', size=self.size, normalized=False, filelist=[1,2,3,4,5])

#         self.labels=labels
#         self.images=images
        


#     def __getitem__(self, index):
#         target=self.labels[index]
#         img=self.images[index]
#         img=GetPhoto(img)
#         return img,target

#     def __len__(self):
#         return len(self.labels)

# class trainset_randomPruning(Dataset):
#     def __init__(self,positive='automobile',negative='truck', size=(224, 224),pruningSize=0.4):
#         self.class_lable_dict={
#             'airplane':0,
#             'automobile':1,
#             'bird':2,
#             'cat':3,
#             'deer':4,
#             'dog':5,
#             'frog':6,
#             'horse':7,
#             'ship':8,
#             'truck':9
#         }
#         self.positive_lable=self.class_lable_dict[positive]
#         self.negative_lable=self.class_lable_dict[negative]
#         self.size=size
#         self.pruningSize=pruningSize

#         labels=getTrainDataByKeyword(keyword='labels', size=self.size, normalized=False, filelist=[1,2,3,4,5])

#         images=getTrainDataByKeyword(keyword='data', size=self.size, normalized=False, filelist=[1,2,3,4,5])

#         self.labels=labels
#         self.images=images

#         self.handle_dataPruning()

#     def _handle_dataPruning(self):
#         length=len(self.data_list)
#         dataToDelete=np.random.randint(length,size=int(length*self.pruningSize))
#         #1是要删的
#         filters=np.zeros(length)
#         for delIndex in dataToDelete:
#             filters[delIndex]=1
        
#         data_list=[]
#         for i,flag in enumerate(filters):
#             if flag==0:
#                 data_list.append(self.data_list[i])
#         self.data_list=data_list[:int(length*(1-self.pruningSize))]



#     def __getitem__(self, index):
#         target=self.labels[index]
#         img=self.images[index]
#         img=GetPhoto(img)
#         return img,target


#     def __len__(self):
#         return len(self.labels)


# class trainset_dataPruning(Dataset):
#     def __init__(self,positive='automobile',negative='truck', size=(224, 224),lossChangePath='',pruningSize=0.5,pos=1):
#         self.class_lable_dict={
#             'airplane':0,
#             'automobile':1,
#             'bird':2,
#             'cat':3,
#             'deer':4,
#             'dog':5,
#             'frog':6,
#             'horse':7,
#             'ship':8,
#             'truck':9
#         }
#         self.positive_lable=self.class_lable_dict[positive]
#         self.negative_lable=self.class_lable_dict[negative]
#         self.size=size
#         self.lossChangePath=lossChangePath
#         self.pruningSize=pruningSize
#         self.pos=pos

#         labels=getTrainDataByKeyword(keyword='labels', size=self.size, normalized=False, filelist=[1,2,3,4,5])

#         images=getTrainDataByKeyword(keyword='data', size=self.size, normalized=False, filelist=[1,2,3,4,5])

#         self.labels=labels
#         self.images=images

#         self.handle_dataPruning()

#     def handle_dataPruning(self):
#         npyPath=self.lossChangePath
#         lossChange_dict=np.load(self.lossChangePath, allow_pickle=True).item()
#         lossChange_dict_sorted=sorted(lossChange_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=False)
#         #找到前一千条
#         if self.pos==1:
#             dataToDelete=np.array(lossChange_dict_sorted)[:int(len(lossChange_dict)*self.pruningSize),[0]].flatten().astype(np.int16)
#         elif self.pos==2:
#             prunNum=int(len(lossChange_dict)*self.pruningSize/2)
#             dataToDelete1=np.array(lossChange_dict_sorted)[:prunNum,[0]].flatten().astype(np.int16)
#             dataToDelete2=np.array(lossChange_dict_sorted)[-prunNum:,[0]].flatten().astype(np.int16)
#             dataToDelete=np.concatenate((dataToDelete1,dataToDelete2))
#         elif self.pos==3:
#             dataToDelete=np.array(lossChange_dict_sorted)[-int(len(lossChange_dict)*self.pruningSize):,[0]].flatten().astype(np.int16)
        
#         #1-10000 变成0-9999
#         dataToDelete=dataToDelete-1

#         #1是要删的
#         filters=np.zeros(len(lossChange_dict))
#         for delIndex in dataToDelete:
#             filters[delIndex]=1
        
#         data_list=[]
#         for i,flag in enumerate(filters):
#             if flag==0:
#                 data_list.append(self.data_list[i])
#         # random.shuffle(data_list)
#         self.data_list=data_list
#         # print(len(self.data_list))
#         # sys.exit()

#     def __getitem__(self, index):
#         target=self.labels[index]
#         img=self.images[index]
#         img=GetPhoto(img)
#         return img,target


#     def __len__(self):
#         return len(self.labels)

# class testset(Dataset):
#     def __init__(self,positive='automobile',negative='truck', size=(224, 224)):
#         self.class_lable_dict={
#             'airplane':0,
#             'automobile':1,
#             'bird':2,
#             'cat':3,
#             'deer':4,
#             'dog':5,
#             'frog':6,
#             'horse':7,
#             'ship':8,
#             'truck':9
#         }
#         self.positive_lable=self.class_lable_dict[positive]
#         self.negative_lable=self.class_lable_dict[negative]
#         self.size=size

#         labels=getTestDataByKeyword(keyword='labels', size=self.size, normalized=False)

#         images=getTestDataByKeyword(keyword='data', size=self.size, normalized=False)

#         self.labels=labels
#         self.images=images

#     def __getitem__(self, index):
#         target=self.labels[index]
#         img=self.images[index]
#         img=GetPhoto(img)
#         return img,target


#     def __len__(self):
#         return len(self.labels)

class CIFAR10_DataPruning(torchvision.datasets.vision.VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            datapruning=False, #要不要数据裁剪
            lossChangePath='',
            pruningSize=0.5,
            pos=1
    ) -> None:

        super(CIFAR10_DataPruning, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.datapruning=datapruning
        self.lossChangePath=lossChangePath
        self.pruningSize=pruningSize
        self.pos=pos

        self.train = train  # training set or test set
        
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.datapruning and self.lossChangePath!='':
            self._handle_dataPruning()
        elif self.datapruning and self.lossChangePath=='':
            self._handle_randomPruning()

        self._load_meta()

    def _handle_randomPruning(self):
        length=len(self.data)
        dataToDelete=np.random.randint(length,size=int(length*self.pruningSize))
        filters=np.zeros(length)
        for delIndex in dataToDelete:
            filters[delIndex]=1
        
        data_list=[]
        for i,flag in enumerate(filters):
            if flag==0:
                data_list.append(self.data_list[i])

        data=[]
        targets=[]

        data_list=[]
        for i,flag in enumerate(filters):
            if flag==0:
                data.append(self.data[i])
                targets.append(self.targets[i])
                
        self.data=data
        self.targets=targets

    def _handle_dataPruning(self):
        npyPath=self.lossChangePath
        lossChange_dict=np.load(self.lossChangePath, allow_pickle=True).item()
        lossChange_dict_sorted=sorted(lossChange_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=False)
        #找到前一千条
        if self.pos==1:
            dataToDelete=np.array(lossChange_dict_sorted)[:int(len(lossChange_dict)*self.pruningSize),[0]].flatten().astype(np.int16)
        elif self.pos==2:
            prunNum=int(len(lossChange_dict)*self.pruningSize/2)
            dataToDelete1=np.array(lossChange_dict_sorted)[:prunNum,[0]].flatten().astype(np.int16)
            dataToDelete2=np.array(lossChange_dict_sorted)[-prunNum:,[0]].flatten().astype(np.int16)
            dataToDelete=np.concatenate((dataToDelete1,dataToDelete2))
        elif self.pos==3:
            dataToDelete=np.array(lossChange_dict_sorted)[-int(len(lossChange_dict)*self.pruningSize):,[0]].flatten().astype(np.int16)
        
        #1-10000 变成0-9999
        dataToDelete=dataToDelete-1

        #1是要删的
        filters=np.zeros(len(lossChange_dict))
        for delIndex in dataToDelete:
            filters[delIndex]=1

        data=[]
        targets=[]

        data_list=[]
        for i,flag in enumerate(filters):
            if flag==0:
                data.append(self.data[i])
                targets.append(self.targets[i])
        self.data=data
        self.targets=targets


    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



if __name__ == "__main__":
    train_data_randomPruning=trainset_randomPruning()
    print(len(train_data_randomPruning))
    # train_data  = trainset()
    # train_dataloader = DataLoader(train_data, batch_size=10,shuffle=False)
    # train_iter=iter(train_dataloader)
    # img,target=train_iter.next()
    # #(10,32,32,3)
    # print(img)
    # print(target)

    # test_data  = testset()
    # test_dataloader = DataLoader(test_data, batch_size=10,shuffle=False)
    # test_iter=iter(test_dataloader)
    # img,target=test_iter.next()
    # #(10,32,32,3)
    # print(img)
    # print(target)