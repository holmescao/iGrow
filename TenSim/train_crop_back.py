# real    0m11.969s
# user    1m0.618s
# sys     0m1.556s
import os
import torch
from utils.model import Net
from utils.data_reader import TomatoDataset
from utils.trainer import data_prepare, train_nn


def mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)


if __name__ == '__main__':
    version = 'result_500'
    new_version = 'teams_v10_team8'
    net = Net(4, 1, 600)
    net.load_state_dict(torch.load('./%s/model/simulator_crop_back.pkl' % version,
                                   map_location=torch.device("cpu")))

    # Team data： train_x, train_y
    train_dir = 'data/team_cache'
    wur_tomato_reader = TomatoDataset(train_dir, max_length=1000)
    train_data, train_data_desc = wur_tomato_reader.read_data(train_dir)
    full_train_x, full_train_y = wur_tomato_reader.data_process(train_data)
    train_x, train_y = wur_tomato_reader.crop_back_x_y(
        full_train_x, full_train_y)

    dealDataset, train_loader, val_loader, test_loader = data_prepare(
        train_x, train_y,
        x_scaler_path='./%s/scaler/crop_back_x_scaler.pkl' % version,
        y_scaler_path='./%s/scaler/crop_back_y_scaler.pkl' % version,
        batch_size=512,
        train_shuffle=True,
        test_shuffle=False)

    save_model_dir = './%s/model/' % new_version
    train_log_dir = './%s/log/' % new_version
    mkdir(save_model_dir)
    mkdir(train_log_dir)
    train_nn(net=net,
             train_loader=train_loader,
             val_loader=val_loader,
             lr=1e-5,
             Epoch=100,
             save_model=save_model_dir+'simulator_crop_back.pkl',
             train_log=train_log_dir+'trainlog_crop_back.log')
