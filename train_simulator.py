import argparse
import os
import torch
from TenSim.utils.model import Net
from TenSim.utils.data_reader import TomatoDataset
from TenSim.utils.trainer import data_prepare, train_nn


number_of_hidden_dims1 = 300
number_of_hidden_dims2 = 300
number_of_hidden_dims3 = 600
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
DAY_IN_LIFE_CYCLE = 160
EPOCHS = 1


def train_greenhouse(wur_tomato_reader, full_train_x, full_train_y, tmp_folder):
    for g_path in [tmp_folder + '/model', tmp_folder + '/scaler', tmp_folder + '/log']:
        if not os.path.exists(g_path):
            os.makedirs(g_path)
    x_scaler_path = tmp_folder + '/scaler/greenhouse_x_scaler.pkl'
    y_scaler_path = tmp_folder + '/scaler/greenhouse_y_scaler.pkl'
    save_model = tmp_folder + '/model/simulator_greenhouse.pkl'
    train_log = tmp_folder + '/log/trainlog_greenhouse.log'
    net = Net(input_dim=13, output_dim=3, hidden_dim=number_of_hidden_dims1)

    train_x, train_y = wur_tomato_reader.greenhouse_x_y(
        full_train_x, full_train_y)

    dealDataset, train_loader, val_loader, test_loader = data_prepare(
        train_x, train_y,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        batch_size=BATCH_SIZE,
        train_shuffle=True,
        test_shuffle=False)

    train_nn(net=net,
             train_loader=train_loader,
             val_loader=val_loader,
             lr=LEARNING_RATE,
             Epoch=EPOCHS,
             save_model=save_model,
             train_log=train_log)


def train_crop_front(wur_tomato_reader, full_train_x, full_train_y, tmp_folder):
    for g_path in [tmp_folder + '/model', tmp_folder + '/scaler', tmp_folder + '/log']:
        if not os.path.exists(g_path):
            os.makedirs(g_path)
    x_scaler_path = tmp_folder + '/scaler/crop_front_x_scaler.pkl'
    y_scaler_path = tmp_folder + '/scaler/crop_front_y_scaler.pkl'
    save_model = tmp_folder + '/model/simulator_crop_front.pkl'
    train_log = tmp_folder + '/log/trainlog_crop_front.log'
    net = Net(input_dim=7, output_dim=3, hidden_dim=number_of_hidden_dims2)

    train_x, train_y = wur_tomato_reader.crop_front_x_y(
        full_train_x, full_train_y)

    dealDataset, train_loader, val_loader, test_loader = data_prepare(
        train_x, train_y,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        batch_size=BATCH_SIZE,
        train_shuffle=True,
        test_shuffle=False)
    train_nn(net=net,
             train_loader=train_loader,
             val_loader=val_loader,
             lr=LEARNING_RATE,
             Epoch=EPOCHS,
             save_model=save_model,
             train_log=train_log)


def train_crop_back(wur_tomato_reader, full_train_x, full_train_y, tmp_folder):
    for g_path in [tmp_folder + '/model', tmp_folder + '/scaler', tmp_folder + '/log']:
        if not os.path.exists(g_path):
            os.makedirs(g_path)
    x_scaler_path = tmp_folder + '/scaler/crop_back_x_scaler.pkl'
    y_scaler_path = tmp_folder + '/scaler/crop_back_y_scaler.pkl'
    save_model = tmp_folder + '/model/simulator_crop_back.pkl'
    train_log = tmp_folder + '/log/trainlog_crop_back.log'

    net = Net(input_dim=4, output_dim=1, hidden_dim=number_of_hidden_dims3)

    train_x, train_y = wur_tomato_reader.crop_back_x_y(
        full_train_x, full_train_y)

    dealDataset, train_loader, val_loader, test_loader = data_prepare(
        train_x, train_y,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        batch_size=BATCH_SIZE,
        train_shuffle=True,
        test_shuffle=False)
    train_nn(net=net,
             train_loader=train_loader,
             val_loader=val_loader,
             lr=LEARNING_RATE,
             Epoch=EPOCHS,
             save_model=save_model,
             train_log=train_log)


def train_model(args):
    traj_train_files = os.path.join(
        args.base_input_path, args.traj_train_files)
    tmp_folder = os.path.join(args.model_dir, args.version)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    wur_tomato_reader = TomatoDataset(
        train_file=traj_train_files, tmp_folder=tmp_folder)
    train_data = wur_tomato_reader.read_data(traj_train_files)
    full_train_x, full_train_y = wur_tomato_reader.data_process(train_data)

    print("train simulator:")
    print('start greenhouse model training')
    train_greenhouse(wur_tomato_reader, full_train_x, full_train_y, tmp_folder)
    print('end greenhouse model training')
    print('--------------------------------')
    print('start front crop model training')
    train_crop_front(wur_tomato_reader, full_train_x, full_train_y, tmp_folder)
    print('end front crop model training')
    print('--------------------------------')
    print('start back crop model training')
    train_crop_back(wur_tomato_reader, full_train_x, full_train_y, tmp_folder)
    print('end back crop model training')
    print('--------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_input_path", default="./input", type=str)
    parser.add_argument(
        "--model_dir", default="./result/models_new/", type=str)
    parser.add_argument("--traj_train_files",
                        default="test-sim.txt", type=str)
    parser.add_argument("--version", default="baseline", type=str)
    args = parser.parse_args()

    train_model(args)
