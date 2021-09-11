import datetime

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
# from utils.model import Net
# from utils.data_reader import TomatoDataset
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.set_num_threads(1)


class DealDataset(Dataset):
    """


    """

    def __init__(self, train_x, train_y, x_scaler_path, y_scaler_path):
        self.x_data, self.y_data, self.x_scaler, self.y_scaler = self.get_data(train_x, train_y, x_scaler_path,
                                                                               y_scaler_path)
        self.len = len(self.x_data)

    def get_data(self, train_x, train_y, x_scaler_path, y_scaler_path):
        '''  

        x_scaler = pickle.load(open(x_scaler_path, 'rb'))
        y_scaler = pickle.load(open(y_scaler_path, 'rb'))

        data_x_normal = x_scaler.transform(train_x)
        data_y_normal = y_scaler.transform(train_y)
        '''

        data_x_normal, x_scaler = self.normalization(train_x)
        data_y_normal, y_scaler = self.normalization(train_y)

        if not os.path.exists(x_scaler_path):
            pickle.dump(x_scaler, open(x_scaler_path, 'wb'))
            pickle.dump(y_scaler, open(y_scaler_path, 'wb'))

        return data_x_normal, data_y_normal, x_scaler, y_scaler

    def normalization(self, data):
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        data_normal = scaler.fit_transform(data)
        return data_normal, scaler

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean(0))


def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.datetime.now()) + ': ' + s + '\n')
    f.close()


def idMapTarjectory(indices, period):
    indices_length = len(indices)
    traj_ids = np.zeros(indices_length*period)
    for id in range(indices_length):
        traj_ids[id*period:(id+1)*period] = np.arange(indices[id]
                                                      * period, (indices[id]+1) * period)

    return traj_ids.astype(np.int32)


def data_prepare_old(train_x,
                     train_y,
                     x_scaler_path,
                     y_scaler_path,
                     batch_size,
                     train_val_size,
                     val_ratio,
                     test_size,
                     period,
                     train_shuffle,
                     test_shuffle
                     ):

    trainval_batch_size = batch_size * period
    test_batch_size = 1 * period

    dealDataset = DealDataset(train_x, train_y, x_scaler_path, y_scaler_path)
    print(f"hours: {len(dealDataset)}")

    total_trajectory = train_val_size + test_size
    split = [test_size]
    split.append(int(np.floor(train_val_size * val_ratio))+test_size)

    traj_indices = list(range(total_trajectory))
    train_traj_ids = traj_indices[split[1]:]
    val_traj_ids = traj_indices[split[0]:split[1]]
    test_traj_ids = traj_indices[:split[0]]

    train_idx = idMapTarjectory(train_traj_ids, period)
    val_idx = idMapTarjectory(val_traj_ids, period)
    test_idx = idMapTarjectory(test_traj_ids, period)

    train_set = Subset(dealDataset, train_idx)
    val_set = Subset(dealDataset, val_idx)
    test_set = Subset(dealDataset, test_idx)

    # x_scaler, y_scaler = dealDataset.x_scaler, dealDataset.y_scaler

    train_loader = DataLoader(dataset=train_set,
                              batch_size=trainval_batch_size,
                              shuffle=train_shuffle)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=trainval_batch_size,
                            shuffle=train_shuffle)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=test_batch_size,
                             shuffle=test_shuffle)

    return dealDataset, train_loader, val_loader, test_loader


def data_prepare(train_x,
                 train_y,
                 x_scaler_path,
                 y_scaler_path,
                 batch_size=64,
                 train_shuffle=True,
                 test_shuffle=False
                 ):
 
    dealDataset = DealDataset(train_x, train_y, x_scaler_path, y_scaler_path)
    print(len(dealDataset))
    ratio = [7, 2, 1]
    length = [len(dealDataset) * ratio[0] // 10,
              len(dealDataset) * ratio[1] // 10]
    length.append(len(dealDataset) - length[0] - length[1])
    print('train: validation : test length', length)
    train_db, val_db, test_db = torch.utils.data.random_split(
        dealDataset, [length[0], length[1], length[2]])
    x_scaler, y_scaler = dealDataset.x_scaler, dealDataset.y_scaler
    #
    train_loader = DataLoader(dataset=train_db,
                              batch_size=batch_size,
                              shuffle=train_shuffle)
    val_loader = DataLoader(dataset=val_db,
                            batch_size=batch_size,
                            shuffle=train_shuffle)
    test_loader = DataLoader(dataset=test_db,
                             batch_size=1,
                             shuffle=test_shuffle)

    return dealDataset, train_loader, val_loader, test_loader


def train_nn(net,
             train_loader,
             val_loader,
             lr=0.001,
             Epoch=13,
             save_model='../model/simulator.pkl',
             train_log='./trainlog.log'):

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss(reduce=True, reduction='sum')  # sum loss
    # mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                            milestones=list(
    #                                                                range(50, Epoch, 50)),
    #                                                            gamma=0.8)
    mult_step_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.8)

    train_loss = []
    valid_loss = []
    min_valid_loss = 10000
    # min_train_loss = 100
    net.to(device)
    for i, epoch in enumerate(range(Epoch)):
        total_train_loss = []

        net.train()
        for j, data in enumerate(train_loader):
            x, y = data
            # x, y = Variable(x).float(), Variable(y).float()
            x, y = torch.FloatTensor(x.float()).to(
                device), torch.FloatTensor(y.float()).to(device)
            prediction = net(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            total_train_loss.append(loss)
            # total_train_loss[j] = loss
        train_loss.append(np.mean(total_train_loss))
        # train_loss.append(torch.mean(total_train_loss).detach().cpu())

        #################################eval ###############################
        total_valid_loss = []
        # total_valid_loss = torch.zeros(len(val_loader))
        net.eval()
        for step, (b_x, b_y) in enumerate(val_loader):
            b_x = torch.FloatTensor(b_x.float()).to(device)
            b_y = torch.FloatTensor(b_y.float()).to(device)
            pred = net(b_x)
            loss = loss_func(pred, b_y)
            loss = loss.detach().item()
            total_valid_loss.append(loss)
            # total_valid_loss[step] = loss.item()
        valid_loss.append(np.mean(total_valid_loss))
        # valid_loss.append(torch.mean(total_valid_loss).detach().cpu())

        lr = optimizer.param_groups[0]['lr']

        if (valid_loss[-1] <= min_valid_loss):
            print('epoch:{}, save!'.format(epoch))

            torch.save(net.state_dict(), save_model)
            min_valid_loss = valid_loss[-1]

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.10f}, valid_loss: {:0.10f}, '
                      'best_valid_loss: {:0.10f}, lr: {:0.10f}').format((i + 1), Epoch,
                                                                        train_loss[-1],
                                                                        valid_loss[-1],
                                                                        min_valid_loss,
                                                                        lr)
        mult_step_scheduler.step()

        print(str(datetime.datetime.now()) + ': ')
        print(log_string)
        log(train_log, log_string)

        ###############################################################################

        # lr = optimizer.param_groups[0]['lr']

        # if (train_loss[-1] <= min_train_loss):
        #     #print('epoch:{}, save!'.format(epoch))
        #

        #     torch.save(net.state_dict(), save_model)
        #     print("save_model")
        #     min_train_loss = train_loss[-1]

        # log_string = ('iter: [{:d}/{:d}], train_loss: {:0.10f}, '
        #               'best_train_loss: {:0.10f}, lr: {:0.10f}').format((i + 1), Epoch,
        #                                                                 train_loss[-1],
        #                                                                 min_train_loss,
        #                                                                 lr)
        # mult_step_scheduler.step()

        # print(str(datetime.datetime.now()) + ': ')
        # print(log_string)
        # log(train_log, log_string)
