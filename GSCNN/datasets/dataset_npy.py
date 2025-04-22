from torch.utils.data import Dataset
import numpy as np



class MyDataSet(Dataset):
    """自定义数据集"""

    # 2023616 添加SST数据集
    # def __init__(self, root,transform):
    def __init__(self, data, transform=None):
        self.data = data

        #  使用新的数据集
        edge_train_data = np.expand_dims(np.load('./datasets/edge_train_data.npy'), 3)
        SSH_train = np.expand_dims(np.load('./datasets/train_img_data.npy'), 3)
        # todo 这里可能需要处理数据，把sst和ssh分开
        SST_train = np.expand_dims(np.load('./datasets/train_img_data.npy'), 3)
        # SSH_train_bak = np.expand_dims(np.load('/datasets/train_img_data_bak.npy'),3)
        Seg_train = np.expand_dims(np.load('./datasets/seg_train_data.npy'), 3)

        seg_train = np.eye(3)[Seg_train[:, :, :, 0]]
        edge2 = np.eye(3)[edge_train_data[:, :, :, 0]]

        self.input = SSH_train
        self.input2 = SST_train
        # self.SSH_train_bak = SSH_train_bak
        self.edge = edge_train_data
        self.mask = seg_train
        self.edge2 = edge2
        # self.images_path = torch.tensor(np.load(os.path.join(data, "train.npy")))
        # self.images_class = torch.tensor(np.load(os.path.join(label, "label.npy")))
        self.transform = transform

    def __len__(self):
        return self.input.shape[0]  # 返回数据的总个数

    def __getitem__(self, index):
        input = self.input[index, :, :]  # 读取每一个npy的数据
        input2 = self.input2[index, :, :]  # 读取每一个npy的数据
        edge = self.edge[index, :, :]  # 读取每一个npy的数据
        mask = self.mask[index, :, :]  # 读取每一个npy的数据
        edge2 = self.edge2[index, :, :]  # 读取每一个npy的数据

        if self.transform is not None:
            input = self.transform(input)
            input2 = self.transform(input2)

        # return input,input2, mask,edge,edge2  # 返回数据还有标签
        return input, input2, mask, edge  # 返回数据还有标签


class valDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        # edge_test_data = np.load('/home/eddy/GSCNN-master/datasets/edge_test_data.npy')
        # # datalength\heigh weith channel
        # SSH_test = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/test_img_data.npy'), 3)
        # Seg_test = np.expand_dims(np.load('/home/eddy/GSCNN-master/datasets/seg_test_data.npy'), 3)
        # 使用新的数据
        edge_test_data = np.load('./datasets/edge_train_data.npy')
        # datalength\heigh weith channel
        SSH_test = np.expand_dims(np.load('./datasets/train_img_data.npy'), 3)
        SST_test = np.expand_dims(np.load('./datasets/train_img_data.npy'), 3)
        Seg_test = np.expand_dims(np.load('./datasets/seg_train_data.npy'), 3)

        seg_test = np.eye(3)[Seg_test[:, :, :, 0]]
        seg_test_miou = Seg_test[:, :, :, 0]

        self.input = SSH_test
        self.input2 = SST_test
        self.mask = seg_test
        self.edge = edge_test_data
        self.seg_test_miou = seg_test_miou

        self.transform = transform

    def __len__(self):
        return self.input.shape[0]  # 返回数据的总个数

    def __getitem__(self, index):
        input = self.input[index, :, :]  # 读取每一个npy的数据
        input2 = self.input2[index, :, :]  # 读取每一个npy的数据
        edge = self.edge[index, :, :]  # 读取每一个npy的数据
        mask = self.mask[index, :, :, :]  # 读取每一个npy的数据
        seg_test_miou = self.seg_test_miou[index, :, :]  # 读取每一个npy的数据

        # 做数据增强的方式
        if self.transform is not None:
            input = self.transform(input)
            input2 = self.transform(input2)
        # return input,input2, seg_test_miou ,edge  # 返回数据还有标签
        # return input,input2, mask ,edge,seg_test_miou  # 返回数据还有标签
        return input, input2, mask, edge  # 返回数据还有标签
