import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import math



class BDGP(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha):
        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].astype(np.int32).reshape(2500,)
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        data1 = min_max_scaler.fit_transform(data1)
        data2 = min_max_scaler.fit_transform(data2)
        self.X = [data1,data2]
        self.Y = labels
        self.user_data = split_data(num_user,Dirichlet_alpha,self.Y)

    def __len__(self):
        return self.X[0].shape[0]

class NUSWIDE(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha):
        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        data1 = scipy.io.loadmat(path+'NUSWIDE.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'NUSWIDE.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X5'].astype(np.float32)
        labels = scipy.io.loadmat(path+'NUSWIDE.mat')['Y'].astype(np.int32).reshape(5000,)
        rep_mapping = {14: 0, 19: 1, 23: 2, 28: 3, 29: 4}
        for i in range(len(labels)):
            idy = rep_mapping.get(labels[i])
            labels[i] = idy
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        data1 = min_max_scaler.fit_transform(data1)
        data2 = min_max_scaler.fit_transform(data2)
        data3 = min_max_scaler.fit_transform(data3)
        data4 = min_max_scaler.fit_transform(data4)
        data5 = min_max_scaler.fit_transform(data5)
        self.X = [data1,data2,data3,data4,data5]
        self.Y = labels
        self.user_data = split_data(num_user,Dirichlet_alpha,self.Y)

    def __len__(self):
        return self.X[0].shape[0]


class MNIST_USPS(Dataset):
    def \
            __init__(self, path, num_user, Dirichlet_alpha):
        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        data1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32).reshape(5000,-1)
        data2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32).reshape(5000,-1)
        self.X = [data1, data2]
        self.user_data = split_data(num_user,Dirichlet_alpha,self.Y)
    def __len__(self):
        return 5000


class Fashion(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha):
        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        data1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)
        self.X = [data1, data2, data3]
        self.user_data = split_data(num_user,Dirichlet_alpha,self.Y)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def split_data(num_user, Dirichlet_alpha, Y):
    dict_users = {i: np.array([]) for i in range(num_user)}
    N = len(Y)
    n_classes = max(Y) + 1

    min_size = 0
    min_require_size = 10

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_user)]
        for k in range(n_classes):
            idx_k = np.where(Y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(Dirichlet_alpha, num_user))
            proportions = np.array(
                [p * (len(idx_j) < (N / num_user * 2)) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_user):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]
    return dict_users

class DatasetSplit(Dataset):
  """An abstract Dataset class wrapped around Pytorch Dataset class."""

  def __init__(self, dataset_x, dataset_y, idxs, dims, num_views):
    self.dataset_x = []
    self.view = len(dims)
    zero_view = []
    for view in range(self.view):
        if view in num_views:
            self.dataset_x.append(dataset_x[view][idxs])
        else:
            zero_view.append(view)
            self.dataset_x.append(np.zeros((len(idxs), dims[view])))
    self.dataset_y = dataset_y[idxs]
    self.idxs = [int(i) for i in idxs]
    self.dims = dims

  def __len__(self):
    return len(self.idxs)

  def __getitem__(self, item):
    image = []
    label = self.dataset_y[item]
    for view in range(self.view):
        x = self.dataset_x[view][item].reshape(self.dims[view])
        image.append(torch.tensor(x))
    return image, torch.tensor(label)



def load_data(dataset,num_user, Dirichlet_alpha):
    if dataset == "BDGP":
        dataset = BDGP('./data/',num_user, Dirichlet_alpha)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5

    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/',num_user, Dirichlet_alpha)
        dims = [784, 784]
        view = 2
        data_size = 5000
        class_num = 10

    elif dataset == "Fashion":
        dataset = Fashion('./data/',num_user, Dirichlet_alpha)
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10

    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('./data/',num_user, Dirichlet_alpha)
        dims = [65, 226, 145, 74, 129]
        view = 5
        data_size = 5000
        class_num = 5
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

def get_mask(view, num_users,missing_rate):
    num_views = []
    assert view >= 2
    miss_client_num = math.floor(num_users * missing_rate)
    num_view = [i for i in range(view)]
    for j in range(num_users):
        if j+miss_client_num < num_users:
            num_views.append(num_view)
        else:
            while True:
                rand_v = np.random.rand(view)
                v_threshold = np.random.rand(1)
                observed_ind = (rand_v >= v_threshold)
                ind_ = ~observed_ind
                rand_v[observed_ind] = 1
                rand_v[ind_] = 0
                if np.sum(rand_v) == 1:
                    break
            num_view = [x for x in num_view if rand_v[x] != 0]
            num_views.append(num_view)
            num_view = [i for i in range(view)]

    return num_views