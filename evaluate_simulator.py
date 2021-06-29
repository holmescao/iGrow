import pickle
import pandas as pd
import numpy as np
import argparse
import os
import warnings
import torch
import gym
from sklearn.metrics import r2_score

from TenSim.utils.data_reader import TomatoDataset
from TenSim.simulator_gpu import PredictModel
from utils.common import mkdir

warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
gym.logger.set_level(40)
torch.set_num_threads(1)


def env(version, base_tmp_folder):
    direcrory = base_tmp_folder+'/models/'
    model_dir = direcrory + version
    model_path = model_dir + '/model/'
    scaler_dir = model_dir + '/scaler/'

    ten_env = PredictModel(model1_dir=model_path+'simulator_greenhouse.pkl',
                           model2_dir=model_path+'simulator_crop_front.pkl',
                           model3_dir=model_path+'simulator_crop_back.pkl',
                           scaler1_x=scaler_dir+'greenhouse_x_scaler.pkl',
                           scaler1_y=scaler_dir+'greenhouse_y_scaler.pkl',
                           scaler2_x=scaler_dir+'crop_front_x_scaler.pkl',
                           scaler2_y=scaler_dir+'crop_front_y_scaler.pkl',
                           scaler3_x=scaler_dir+'crop_back_x_scaler.pkl',
                           scaler3_y=scaler_dir+'crop_back_y_scaler.pkl',
                           linreg_dir=model_path+'/PARsensor_regression_paramsters.pkl',
                           weather_dir=model_path+'/weather.npy')

    return ten_env


def Table3(args):
    print("=============Table3===============")
    tmp_folder = os.path.join(args.base_tmp_folder,
                              'models/%s' % args.model_version)
    wur_tomato_reader = TomatoDataset(args.traj_test_files, tmp_folder)
    train_data = wur_tomato_reader.read_data(args.traj_test_files)
    full_train_x, _ = wur_tomato_reader.data_process(train_data)
    simulator = env(args.model_version, args.base_tmp_folder)

    # PAR model
    PAR_model_path = os.path.join(
        tmp_folder, 'model/PARsensor_regression_paramsters.pkl')
    linreg = pickle.load(open(PAR_model_path, 'rb'))

    columns = ['PAR', 'AirT', 'AirRh', 'AirCO2',
               'LAI', 'PlantLoad', ' NetGrouwth', 'FW']
    save_dir = args.base_tmp_folder+'/table3/'
    mkdir(save_dir)

    save_path = save_dir+'R2_of_per_cache.csv'
    if os.path.exists(save_path):
        os.remove(save_path)

    PAR_R2 = []
    AirT_R2 = []
    AirRH_R2 = []
    Airppm_R2 = []
    LAI_R2 = []
    PlantLoad_R2 = []
    NetGrowth_R2 = []
    FW_R2 = []
    score = []
    for idx in range(len(full_train_x)):
        PAR_list, real_PAR_list = [], []
        AirT_list, real_AirT_list = [], []
        AirRH_list, real_AirRH_list = [], []
        Airppm_list, real_Airppm_list = [], []
        LAI_list, real_LAI_list = [], []
        PlantLoad_list, real_PlantLoad_list = [], []
        NetGrowth_list, real_NetGrowth_list = [], []
        FW_list, real_FW_list = [], []
        input = full_train_x[idx]
        done = False
        simulator.reset()
        for i in range(args.DAY_IN_LIFE_CYCLE):
            control = input[i * 24: (i + 1) * 24, 6: 10]
            control = control.T.reshape(1, -1)[0]
            obs, _, done, _ = simulator.step(control)
            if done:
                break

            for h in range(24):
                par_x = input[i*24 + h: i*24 + h+1, [0, 8]].reshape(1, -1)
                PARsensor = linreg.predict(par_x)
                PARsensor = float(PARsensor) if PARsensor > 50.0 else 0.0
                PAR_list.append(PARsensor)
            real_PAR_list.extend(input[i*24: (i+1)*24, 13])

            AirT_list.extend(obs[: 24])
            real_AirT_list.extend(
                input[i * 24:(i + 1) * 24, 10].reshape(1, -1).tolist()[0])
            AirRH_list.extend(obs[24:48])
            real_AirRH_list.extend(
                input[i * 24:(i + 1) * 24, 11].reshape(1, -1).tolist()[0])
            Airppm_list.extend(obs[48:72])
            real_Airppm_list.extend(
                input[i * 24:(i + 1) * 24, 12].reshape(1, -1).tolist()[0])

            LAI_list.append(obs[72])
            real_LAI_list.append(input[i * 24 + 23, 14])
            PlantLoad_list.append(obs[73])
            real_PlantLoad_list.append(input[i * 24 + 23, 15])
            NetGrowth_list.append(obs[74])
            real_NetGrowth_list.append(input[i * 24 + 23, 16])
            FW_list.append(obs[75])
            real_FW_list.append(input[i * 24 + 23, 17])

        # calculate R^2
        r2_PAR = r2_score(real_PAR_list, PAR_list)
        r2_AirT = r2_score(real_AirT_list, AirT_list)
        r2_AirRH = r2_score(real_AirRH_list, AirRH_list)
        r2_Airppm = r2_score(real_Airppm_list, Airppm_list)
        r2_LAI = r2_score(real_LAI_list, LAI_list)
        r2_PlantLoad = r2_score(real_PlantLoad_list, PlantLoad_list)
        r2_NetGrowth = r2_score(real_NetGrowth_list, NetGrowth_list)
        r2_FW = r2_score(real_FW_list, FW_list)
        goodness = [r2_PAR, r2_AirT, r2_AirRH, r2_Airppm,
                    r2_LAI, r2_PlantLoad, r2_NetGrowth,
                    r2_FW]
        mean_r2 = np.mean(goodness)
        goodness.append(mean_r2)
        print("%d cache score: %.2f" % (idx, mean_r2))

        # # save
        df = pd.DataFrame([goodness], columns=columns+['score'])
        if os.path.exists(save_path):
            ori_df = pd.read_csv(save_path)
            df = ori_df.append(df)
        df.to_csv(save_path, float_format='%.2f', index=False)

        # net1
        PAR_R2.append(r2_PAR)
        AirT_R2.append(r2_AirT)
        AirRH_R2.append(r2_AirRH)
        Airppm_R2.append(r2_Airppm)

        # net2
        LAI_R2.append(r2_LAI)
        PlantLoad_R2.append(r2_PlantLoad)
        NetGrowth_R2.append(r2_NetGrowth)

        # net3
        FW_R2.append(r2_FW)

        score.append(mean_r2)

    # mean
    mean_PAR = np.mean(PAR_R2)
    mean_AirT = np.mean(AirT_R2)
    mean_AirRH = np.mean(AirRH_R2)
    mean_Airppm = np.mean(Airppm_R2)
    mean_LAI = np.mean(LAI_R2)
    mean_PlantLoad = np.mean(PlantLoad_R2)
    mean_NetGrowth = np.mean(NetGrowth_R2)
    mean_FW = np.mean(FW_R2)
    mean_score = np.mean(score)

    goodness_of_simulator = [mean_PAR, mean_AirT, mean_AirRH, mean_Airppm,
                             mean_LAI, mean_PlantLoad, mean_NetGrowth,
                             mean_FW, mean_score]

    # save
    Table3_df = pd.DataFrame([goodness_of_simulator],
                             columns=columns+['score'])
    Table3_df.to_csv(save_dir+'R2_of_simulator.csv',
                     float_format='%.2f', index=False)
    print("mean R2:")
    print(Table3_df.mean(axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tmp_folder", default="./result", type=str)
    parser.add_argument("--model_version", default="incremental", type=str)
    parser.add_argument("--traj_test_files",
                        default="./input/test.txt", type=str)
    parser.add_argument("--DAY_IN_LIFE_CYCLE",
                        default=160, type=int)
    args = parser.parse_args()

    Table3(args)
