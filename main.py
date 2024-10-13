import math

import torch
import os
import scipy.io as sio
import metric
from my_network import Network
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
import copy
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from dataloader import load_data, DatasetSplit, get_mask
from loss import Loss
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


os.environ['OMP_NUM_THREADS'] = '1'
import warnings

# ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)




# MNIST-USPS
# BDGP
# Fashion
# NUSWIDE
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='MNIST-USPS')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=250) # pre-training rounds
parser.add_argument("--main_epochs", default=25) # local training rounds
parser.add_argument("--feature_dim", default=20) # d_m and d
parser.add_argument("--num_users", default=24) # number of clients
parser.add_argument("--Dirichlet_alpha", default=9999)
parser.add_argument("--interval_epoch", default=25)
parser.add_argument("--M_S", default=1) # Multi-view clients / Single-view clients 2:1--2/ 1:1--1/ 1:2--0.5
parser.add_argument("--participate", default=1) # client participation rates

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

if args.dataset == "MNIST-USPS":
    args.num_users = 24
    args.main_epochs = 25
    args.interval_epoch = 25
    seed = 10
if args.dataset == "NUSWIDE":
    args.num_users = 24
    args.main_epochs = 25
    args.interval_epoch = 25
    seed = 10
if args.dataset == "BDGP":
    args.num_users = 12
    args.main_epochs = 10
    args.interval_epoch = 10
    seed = 10
if args.dataset == "Fashion":
    args.num_users = 48
    args.main_epochs = 25
    args.interval_epoch = 25
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain(nu, model):
    model.train()
    mes = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for pre_epoch in range(args.mse_epochs):
        tot_loss = 0.
        for batch_idx, (xs, ys) in enumerate(data_loader_list[nu]):
            for v in range(view):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            optimizer.zero_grad()
            xrs, q, h, rs = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

    h_list = []
    ys_list = []
    for batch_idx, (xs, ys) in enumerate(data_loader_list[nu]):
        for v in range(view):
            xs[v] = xs[v].to(device)
            xs[v] = xs[v].to(torch.float32)
        _, _, h, _ = model(xs)
        h_list.append(h)
        ys_list.append(ys)
    h_list = torch.cat(h_list, dim=0)
    ys_list = torch.cat(ys_list, dim=0)
    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
    kmeans.fit(h_list.detach().cpu().numpy())
    labels = kmeans.predict(h_list.detach().cpu().numpy())
    print('client',nu,'pretrain acc', compute_acc(labels, ys_list.detach().cpu().numpy()))
    cluster_centers = kmeans.cluster_centers_
    model.centroids.data = copy.deepcopy(torch.tensor(cluster_centers)).cuda()



def local_train(nu, model, glob_model, isfull):
    model.train()
    mes = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    for epoch in range(1):
        tot_loss = 0.
        for batch_idx, (xs, ys) in enumerate(data_loader_list[nu]):
            for v in range(view):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            optimizer.zero_grad()
            xrs, zs, h, rs = model(xs)
            _, glob_z, glob_h, glob_r = glob_model(xs)
            loss_list = []
            for v in range(view):
                if isfull:
                    loss_list.append(criterion.forward_feature(rs[v],h))
                else:
                        if v in num_views[nu]:
                            loss_list.append(criterion.forward_model(glob_h, h, zs[v]))
                loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

def local_full_train(nu, local_model, glob_models_temp):
    glob_models_weights = copy.deepcopy(glob_models_temp)
    for key, model in glob_models_temp.items():
        model.train()
        num_view = list(key)
        mes = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for epoch in range(50):
            tot_loss = 0.
            for batch_idx, (xs, ys) in enumerate(data_loader_list[nu]):
                xs_missing = copy.deepcopy(xs)
                for v in range(view):
                    xs[v] = xs[v].to(device)
                    xs[v] = xs[v].to(torch.float32)
                    if v not in num_view:
                        xs_missing[v] = torch.tensor(np.zeros((len(xs_missing[v]), dims[v])))
                    xs_missing[v] = xs_missing[v].to(device)
                    xs_missing[v] = xs_missing[v].to(torch.float32)
                optimizer.zero_grad()
                _, _, h_ref, _ = local_model(xs)
                xrs, zs, h, rs = model(xs_missing)
                if epoch==0:
                    _, _, glob_h, _ = model(xs_missing)
                loss_list = []
                for v in range(view):
                    loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(mes(h, h_ref))

                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()

        mi = 0
        for v in range(view):
            if v in num_view:
                if num_view.__len__() < view:
                    mi = metric.mutual_information(h, glob_h) - metric.mutual_information(h, zs[num_view[0]])
                else:
                    mi += metric.mutual_information(h, rs[v])

        glob_models_temp[key] = model
        glob_models_weights[key] = mi

    return glob_models_temp, glob_models_weights

def local_single_train(nu, local_model, glob_model, glob_models_temp):
    glob_models_weights = copy.deepcopy(glob_models_temp)
    for key, model in glob_models_temp.items():
        model.eval()
        num_view = list(key)
        mi = 0
        if num_views[nu] == num_view:
            for batch_idx, (xs, ys) in enumerate(data_loader_list[nu]):
                for v in range(view):
                    xs[v] = xs[v].to(device)
                    xs[v] = xs[v].to(torch.float32)

                xrs, zs, h, rs = local_model(xs)
                _, glob_z, glob_h, glob_r = glob_model(xs)
            mi = metric.mutual_information(h, glob_h) - metric.mutual_information(h, zs[num_view[0]])


            glob_models_temp[key] = local_model
        glob_models_weights[key] = mi

    return glob_models_temp, glob_models_weights



def valid_global(model, valid_dataset_list):
    local_hs, local_ys = [], []
    for an in range(args.num_users):
        h_list, ys_list = [], []
        model.eval()
        for batch_idx, (xs, ys) in enumerate(valid_dataset_list[an]):
            for v in range(len(xs)):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            xrs, _, h, rs  = model(xs)
            h_list.append(h)
            ys_list.append(ys)
        local_hs.append(torch.cat(h_list, dim=0))
        local_ys.append(torch.cat(ys_list, dim=0))

    global_hs = torch.cat(local_hs, dim=0)
    global_ys = torch.cat(local_ys, dim=0)


    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
    kmeans.fit(global_hs.detach().cpu().numpy())
    labels = kmeans.predict(global_hs.detach().cpu().numpy())
    print('global model acc', compute_acc(labels, global_ys.detach().cpu().numpy()))


def valid_client(model, valid_dataset, nu):
    h_list, ys_list, rs_list = [], [], []
    model.eval()
    for batch_idx, (xs, ys) in enumerate(valid_dataset):
        for v in range(len(xs)):
            xs[v] = xs[v].to(device)
            xs[v] = xs[v].to(torch.float32)
        xrs, _, h, rs  = model(xs)
        rss = []
        for v in range(len(xs)):
            if v in num_views[nu]:
                rss.append(rs[v])
        rss = torch.cat(rss, dim=1)
        rs_list.append(rss)
        h_list.append(h)
        ys_list.append(ys)

    local_hs = torch.cat(h_list, dim=0)
    local_ys = torch.cat(ys_list, dim=0)

    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
    kmeans.fit(local_hs.detach().cpu().numpy())
    labels = kmeans.predict(local_hs.detach().cpu().numpy())
    print('client',nu,'acc', compute_acc(labels,local_ys.detach().cpu().numpy()))


def valid(valid_model_list, valid_dataset_list,valid_users):
    local_hs, local_ys, local_labels = [], [], []
    init_center = []
    for an in range(valid_users):
        h_list, ys_list= [], []
        valid_model_list[an].eval()
        for batch_idx, (xs, ys) in enumerate(valid_dataset_list[an]):
            for v in range(len(xs)):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            xrs, _, h, rs  = valid_model_list[an](xs)
            h_list.append(h)
            ys_list.append(ys)
        local_hs = torch.cat(h_list, dim=0)
        local_ys.append(torch.cat(ys_list, dim=0))

        if an == 0:
            kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
            kmeans.fit(local_hs.detach().cpu().numpy())
            init_center = kmeans.cluster_centers_
        else:
            kmeans = KMeans(n_clusters=class_num, init=init_center)
            kmeans.fit(local_hs.detach().cpu().numpy())
        labels = kmeans.predict(local_hs.detach().cpu().numpy())
        local_labels.append(labels)

    global_ys = torch.cat(local_ys, dim=0)
    global_labels = np.concatenate(local_labels, axis=0)

    test_acc = compute_acc(global_labels, global_ys.detach().cpu().numpy())
    test_nmi = normalized_mutual_info_score(global_labels, global_ys.detach().cpu().numpy())
    test_ari = adjusted_rand_score(global_labels, global_ys.detach().cpu().numpy())
    print('overall acc', test_acc)
    print('overall nmi', test_nmi)
    print('overall ari', test_ari)

    return test_acc,test_nmi,test_ari


def match(y_pred, y_true):
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):  # 5000
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind[1]


def compute_acc(y_pred, y_true):
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def aggregate_models(model_list, weights=None):
    if weights is None:
        weights = [1.0 / len(model_list)] * len(model_list)
    agg_model = copy.deepcopy(model_list[0])
    agg_state_dict = agg_model.state_dict()
    for key in agg_state_dict:
        agg_state_dict[key].zero_()
    for model, weight in zip(model_list, weights):
        model_state_dict = model.state_dict()
        for key in agg_state_dict:
            agg_state_dict[key] += model_state_dict[key] * weight
    agg_model.load_state_dict(agg_state_dict)
    return agg_model

# The result is a global model list with selective aggregation
def aggregate_models1(model_list):
    # group
    grouped_dict = defaultdict(list)
    for idx, sub_list in enumerate(num_views):
        grouped_dict[tuple(sub_list)].append(idx)

    glob_models = dict()
    for key, value in grouped_dict.items():
        agg_models = []
        for v in value:
            agg_models.append(model_list[v])
        glob_models[key] = aggregate_models(agg_models)


    return grouped_dict, glob_models

# weighting aggregation
def aggregate_models2(model_list, grouped_dict, weights_list):

    glob_models = dict()
    for key, _ in grouped_dict.items():
        agg_models = []
        agg_weights = []
        for nu in range(len(model_list)):
            agg_models.append(model_list[nu][key])
            agg_weights.append(weights_list[nu][key])

        total_sum = np.sum(agg_weights)
        weights = agg_weights / total_sum
        glob_models[key] = aggregate_models(agg_models,weights=weights)


    return glob_models


def save_data(num_users,num_views,data_loader_list,missing_rate):
    len1 = len(data_loader_list[0].dataset.idxs)
    mask = np.ones((len1,view))
    for nu in range(1,num_users):
        len1 = len(data_loader_list[nu].dataset.idxs)
        mask1= []
        for v in range(view):
            if v in num_views[nu]:
                mask_temp = np.ones((len1,1))
            else:
                mask_temp = np.zeros((len1, 1))
            if len(mask1)==0:
                mask1 = mask_temp
            else:
                mask1 = np.hstack((mask1,mask_temp))
        mask = np.vstack((mask,mask1))

    data = {'mask': mask}
    sio.savemat('./mask/'+args.dataset+str(missing_rate)+'.mat', data)




if __name__ == '__main__':
    T = 1 # repeated experiment
    accs = []
    nmis = []
    aris = []

    for i in range(T):

        missing_mapping = {2: 0.3333, 1: 0.5, 0.5: 0.6667}
        missing_rate = missing_mapping.get(args.M_S)

        dataset, dims, view, data_size, class_num = load_data(args.dataset, args.num_users, args.Dirichlet_alpha)

        data_loader_list = []
        test_data_loader_list = []

        num_users = args.num_users
        # num_views = get_mask(view, num_users, missing_rate)
        num_views_glob = get_mask(view, num_users, missing_rate)

        for j in range(num_users):
            data_loader = DataLoader(DatasetSplit(dataset.X, dataset.Y, dataset.user_data[j], dims, num_views_glob[j]), batch_size=args.batch_size, shuffle=False)
            data_loader_list.append(copy.deepcopy(data_loader))


        for j in range(num_users):
            test_data_loader = DataLoader(DatasetSplit(dataset.X, dataset.Y, dataset.user_data[j], dims, num_views_glob[j]),
                                     batch_size=data_size, shuffle=False)
            test_data_loader_list.append(copy.deepcopy(test_data_loader))

        # save_data(num_users,num_views_glob,data_loader_list,missing_rate)

        local_models = []

        print('Start Training')
        setup_seed(seed)
        # participate
        num_users = int(args.num_users * args.participate)
        num_full_client = math.ceil(num_users * (1 - missing_rate))
        num_views = num_views_glob[:num_full_client]
        num_full_client_glob = math.ceil(args.num_users * (1 - missing_rate))
        num_views = num_views+num_views_glob[num_full_client_glob:num_full_client_glob+num_users-num_full_client]

        # data_loader_list_participate = data_loader_list[:num_full_client]
        # data_loader_list_participate = data_loader_list_participate + data_loader_list[num_full_client_glob:num_full_client_glob + num_users - num_full_client]
        #
        # num_views_un = num_views_glob[num_full_client:num_full_client_glob]
        # num_views_un = num_views_un+num_views_glob[num_full_client_glob+num_users-num_full_client:]
        # data_loader_list_un = data_loader_list[num_full_client:num_full_client_glob]
        # data_loader_list_un = data_loader_list_un + data_loader_list[num_full_client_glob + num_users - num_full_client:]

        for j in range(num_users):
            local_models.append(copy.deepcopy(Network(view, num_views[j], dims, args.feature_dim, class_num, device).to(device)))

        global_center_list = []
        # pretrain
        for nu in range(num_users):
            pretrain(nu, local_models[nu])
            if nu+1 < num_users:
                local_models[nu+1] = local_models[0]

        global_model = aggregate_models(local_models)
        # valid(local_models, data_loader_list)

        grouped_dict, glob_models = aggregate_models1(local_models)

        # training
        for round in range(5):
            glob_models_list = []
            glob_models_weights_list = []
            for nu in range(num_users):
                isfull = nu<num_full_client
                glob_models_temp = copy.deepcopy(glob_models)
                for me in range(args.main_epochs):
                    if isfull:
                        local_models[nu] = glob_models[tuple(num_views[nu])]
                    local_train(nu, local_models[nu], glob_models[tuple(num_views[nu])], isfull)
                    if me % args.interval_epoch == 0 or me == 0:
                            valid_client(local_models[nu], data_loader_list[nu],nu)

                if isfull:
                    glob_models_temp, glob_models_weights = local_full_train(nu, local_models[nu], glob_models_temp)
                    glob_models_list.append(glob_models_temp)
                    glob_models_weights_list.append(glob_models_weights)

                else:
                    if round>0:
                        glob_models_temp, glob_models_weights = local_single_train(nu, local_models[nu], glob_models[tuple(num_views[nu])], glob_models_temp)
                        glob_models_list.append(glob_models_temp)
                        glob_models_weights_list.append(glob_models_weights)

            glob_models = aggregate_models2(glob_models_list, grouped_dict,glob_models_weights_list)
            glob_models_valid = []
            glob_models_valid_participate = []
            glob_models_valid_unparticipate = []
            for nu in range(args.num_users):
                # glob_models_valid[nu] = copy.deepcopy(glob_models[tuple(num_views[nu])])
                glob_models_valid.append(copy.deepcopy(glob_models[tuple(num_views_glob[nu])]))

            print('----------------round:',round+1)
            # print("client local models")
            # _,_,_=valid(local_models, data_loader_list)
            print("multiple global models")
            test_acc,test_nmi,test_ari = valid(glob_models_valid, data_loader_list,args.num_users)

            # participate
            # for nu in range(num_users):
            #     glob_models_valid_participate.append(copy.deepcopy(glob_models[tuple(num_views[nu])]))
            # print("participating clients")
            # _,_,_=valid(glob_models_valid_participate, data_loader_list_participate, num_users)
            #
            # for nu in range(args.num_users-num_users):
            #     glob_models_valid_unparticipate.append(copy.deepcopy(glob_models[tuple(num_views_un[nu])]))
            # print("unparticipating clients")
            # _,_,_=valid(glob_models_valid_unparticipate, data_loader_list_un, args.num_users-num_users)

        accs.append(test_acc)
        nmis.append(test_nmi)
        aris.append(test_ari)

    # print("----------------Final Results----------------")
    # print("ACC (mean) = {:.4f} ACC (std) = {:.4f}".format(np.mean(accs),
    #                                                           np.std(accs)))
    # print("NMI (mean) = {:.4f} NMI (std) = {:.4f}".format(np.mean(nmis),
    #                                                           np.std(nmis)))
    # print("PUR (mean) = {:.4f} PUR (std) = {:.4f}".format(np.mean(aris),
    #                                                           np.std(aris)))