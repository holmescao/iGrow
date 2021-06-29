# %%
import matplotlib.pyplot as plt
from utils.trainer import data_prepare, train_nn
from utils.data_reader import TomatoDataset
from utils.model import Net
import torch
import os
import argparse
import shutil
from shutil import copyfile


def mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)


def CopyRemainModelToNewVersion(source_path, target_path):
    if not os.path.exists(target_path):
        # 如果目标路径不存在原文件夹的话就创建
        os.makedirs(target_path)

    if os.path.exists(source_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(target_path)

    shutil.copytree(source_path, target_path)


'''作物模型前段训练部分'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--version', default='result500', help='base version')
    parser.add_argument(
        '--modelsDir', default='models', help='dir of all versions models')
    parser.add_argument(
        '--trainDir', default='data/wur_tomato', help='dir of train dataset')
    parser.add_argument(
        '--bs', default=1, help='batch size of trajectory')
    parser.add_argument(
        '--lr', default=1e-5, help='init learning rate')
    parser.add_argument(
        '--Epoch', default=150, help='train epoch')
    parser.add_argument(
        '--trainvalSize', default=800, help='trajectory numbers')
    parser.add_argument(
        '--valRatio', default=0.2, help='ratio of trajectory')
    parser.add_argument(
        '--testSize', default=50, help='trajectory numbers')

    # args = parser.parse_args()
    args = parser.parse_known_args()[0]

    # new version
    args_list = ['@'.join([name, str(val)])
                 for name, val in vars(args).items()]
    new_version = 'net2_'+'_'.join(args_list).replace("/", "-")
    new_version_dir = '/'.join([args.modelsDir, new_version])

    # net
    net = Net(7, 3, 300)
    # net.load_state_dict(torch.load('%s/model/simulator_crop_front.pkl' % version_dir,
    #                                map_location=torch.device("cpu")))

    # dataset
    wur_tomato_reader = TomatoDataset(
        args.trainDir, max_length=args.testSize+args.trainvalSize)
    train_data, data_desc = wur_tomato_reader.read_data(args.trainDir)
    full_train_x, full_train_y = wur_tomato_reader.data_process(train_data)
    # data clip
    full_train_x, full_train_y = wur_tomato_reader.data_clip(
        full_train_x, full_train_y)
    # 提取特征变量
    train_x, train_y = wur_tomato_reader.crop_front_x_y(
        full_train_x, full_train_y)

    # 训练X为7维: AirT, AirRH, Airppm, PARsensor, LAI, PlantLoad, NetGrowth
    # 训练Y为3维: LAI, PlantLoad, NetGrowth

    # plt.plot(train_y[:, 1])
    # plt.show()
# %%
    scaler_dir = new_version_dir + '/scaler'
    mkdir(scaler_dir)
    dealDataset, train_loader, val_loader, test_loader = data_prepare(
        train_x, train_y,
        train_val_size=args.trainvalSize,
        val_ratio=args.valRatio,
        test_size=args.testSize,
        period=3935,
        x_scaler_path='%s/crop_front_x_scaler.pkl' % scaler_dir,
        y_scaler_path='%s/crop_front_y_scaler.pkl' % scaler_dir,
        batch_size=args.bs,
        train_shuffle=False,
        test_shuffle=False)

    # train
    save_model_dir = '%s/' % new_version_dir
    train_log_dir = '%s/log/' % new_version_dir
    mkdir(save_model_dir)
    mkdir(train_log_dir)
    train_nn(net=net,
             train_loader=train_loader,
             val_loader=val_loader,
             lr=args.lr,
             Epoch=args.Epoch,
             save_model=save_model_dir+'simulator_crop_front.pkl',
             train_log=train_log_dir+'trainlog_crop_front.log')

    # test

    '''generate full model in test dir
    source_dir = './models/'
    target_dir = '../models/'
    # copy origin
    source_path = os.path.abspath(source_dir+'%s/' % args.version)
    target_path = os.path.abspath(target_dir+'%s/' % new_version)
    CopyRemainModelToNewVersion(source_path, target_path)
    # copy net2
    copyfile(source_dir+'%s/simulator_crop_front.pkl' % new_version,
             target_dir+'%s/model/simulator_crop_front.pkl' % new_version)
    '''
