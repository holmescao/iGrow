import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy import signal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from TenSim.utils.config import Jinyu_Tomato_v0 as config


class TomatoDataset(object):

    def __init__(self, train_file, tmp_folder):
        self.train_file = train_file
        self.tmp_folder = tmp_folder

    def read_data(self, train_file):
        with open(train_file, 'r') as f:
            train_file_list = f.readlines()

        data = []
        for file_path in train_file_list:
            X = loadmat(file_path.replace('\n', 'X.mat'))['X']
            Y = loadmat(file_path.replace('\n', 'monitor.mat'))['monitor']
            # 如果需要，可以使用如下注释代码来去除表格中的nan值，一般只有Y的第24列MvFogAir会是nan
            # idx = (~np.isnan(Y.sum(0))).nonzero()[0]
            # Y = Y[:, idx]
            data.append((X, Y))
        return data

    def illu_irri_process(self, X):
        '''
        此函数用于将X输入变量列表中的光照和灌溉值从起止时间转变为每个小时是否进行的离散值
        处理完成后，X的第16列代表当前小时的开灯时长，第35列代表当前小时灌溉时长
        '''
        for i in range(164):  # days of each eposide
            # illumination progress
            illu_time = X[i * 24][16]
            illu_end = X[i * 24][17]
            illu_start = illu_end - illu_time
            # irrigation progress
            irri_start = X[i * 24][35]
            irri_end = X[i * 24][36]
            for j in range(24):

                if illu_start - 1 < j < illu_end:
                    if j + 1 - illu_start < 1:
                        X[i * 24 + j][16] = j + 1 - illu_start
                    elif j + 1 > illu_end:
                        X[i * 24 + j][16] = illu_end - j
                    else:
                        X[i * 24 + j][16] = 1
                else:
                    X[i * 24 + j][16] = 0

                if irri_start - 1 < j < irri_end:
                    if j + 1 - irri_start < 1:
                        X[i * 24 + j][35] = j + 1 - irri_start
                    elif j + 1 > irri_end:
                        X[i * 24 + j][35] = irri_end - j
                    else:
                        X[i * 24 + j][35] = 1
                else:
                    X[i * 24 + j][35] = 0

    def data_process(self, data):
        '''
        将所有我们需要的变量按天进行提取
        '''

        # 创建模型所需变量存储路径
        simulator_model_path = os.path.join(self.tmp_folder, 'model')
        if not os.path.exists(simulator_model_path):
            os.makedirs(simulator_model_path)

        if len(data) == 0:
            return []

        HOURS_IN_DAY = 24
        bad_days = [15, 75]

        train_X_list = []
        train_Y_list = []

        for d in data:
            X, Y = d

            '''数据预处理，Y补一维与X长度相等'''
            bad_index = []
            n = X.shape[0]
            for bd in bad_days:
                bad_index += list(range(bd * HOURS_IN_DAY,
                                        bd * HOURS_IN_DAY + HOURS_IN_DAY))
            good_index = sorted(list(set(range(n)).difference(bad_index)))
            X = X[good_index, :]
            Y = np.vstack((Y[0], Y))
            Y = Y[good_index, :]

            # x = Y[:,3:9]
            # np.save('weather.npy', x)
            '''数据预处理结束'''

            '''对开关灯和灌溉维度进行转换'''
            self.illu_irri_process(X)

            '''对NetGrowth变量进行累加操作'''
            Y[:, 37] = np.cumsum(Y[:, 37])
            # 0.0014 * 1e5 /7 *8
            '''对LAI进行平滑'''
            smooth_lai = signal.savgol_filter(
                Y[:, 29], window_length=999, polyorder=2)
            smooth_lai[smooth_lai < 0] = 0
            Y[:, 29] = smooth_lai
            '''对PlantLoad进行平滑'''
            smooth_plantload = signal.savgol_filter(
                Y[:, 35], window_length=999, polyorder=2)
            smooth_plantload[smooth_plantload < 0] = 0
            Y[:, 35] = smooth_plantload

            '''数据选取'''
            outside_weather = Y[:, [3, 4, 5, 6, 7, 8]
                                ]  # Igolb, Tout, RHout, Co2out, Windsp, Tsky
            '''存储天气信息'''
            if not os.path.exists(os.path.join(simulator_model_path, 'weather.npy')):
                np.save(os.path.join(simulator_model_path,
                                     'weather.npy'), outside_weather)
            # comp1.temp, comp1.co2, comp1.illumination, comp1.irrigation
            control = X[:, [19, 23, 16, 35]]
            inside_weather = Y[:, [0, 1, 2]]  # AirT, AirRH, Airppm
            crop_state = Y[:, [29, 35, 37]]  # LAI, PlantLoad, NetGrowth
            fw = Y[:, 31].reshape(len(Y), -1)  # CumFruitsCount

            '''PARsensor linear regression'''
            par_x = np.hstack((outside_weather[:, [0]], control[:, [2]]))
            if not os.path.exists(os.path.join(simulator_model_path, 'PARsensor_regression_paramsters.pkl')):
                par_y = Y[:, 9]
                linreg = LinearRegression()
                linreg.fit(par_x, par_y)
                pickle.dump(linreg,
                            open(os.path.join(simulator_model_path, 'PARsensor_regression_paramsters.pkl'), 'wb'))
            else:
                linreg = pickle.load(
                    open(os.path.join(simulator_model_path, 'PARsensor_regression_paramsters.pkl'), 'rb'))
            PARsensor = linreg.predict(par_x)
            PARsensor = np.where(PARsensor > 50.0, PARsensor, 0)
            PARsensor = PARsensor.reshape(len(Y), -1)

            '''PARsensor值处理结束'''

            '''错位，用当前一天的值去预测下一天的值'''
            train_X = np.hstack(
                (outside_weather, control, inside_weather, PARsensor, crop_state, fw))[:-1]
            train_Y = np.hstack((inside_weather, crop_state, fw))[1:]

            train_X_list.append(train_X)
            train_Y_list.append(train_Y)

        train_X_all = np.array(train_X_list)
        train_Y_all = np.array(train_Y_list)
        """
        X变量列表:   Igolb, Tout, RHout, Co2out, Windsp, Tsky; [0,5]
                    comp1.temp, comp1.co2, comp1.illumination, comp1.irrigation; [6,9]
                    AirT, AirRH, Airppm, PARsensor; [10,13]
                    LAI, PlantLoad, NetGrowth;FW [14,17]
        Y变量列表:   AirT, AirRH, Airppm; 
                    LAI, PlantLoad, NetGrowth;
                    FW
        """
        return train_X_all, train_Y_all

    def PAR_x_y(self, X, Y):
        '''
        此函数用于产生fit PARsensor变量的x和y
        :param X: Iglob, comp1.illumination
        :param Y: PARsensor
        :return:
        '''
        train_X = np.concatenate(X[:, :, [0, 8]], axis=0)
        train_Y = np.concatenate(X[:, :, 13])
        return train_X, train_Y

    def greenhouse_x_y(self, X, Y):
        '''
        此函数用于产生第一阶段即温室模拟阶段的训练数据，
        数据粒度为小时级
        训练X为13维: Igolb, Tout, RHout, Co2out, Windsp, Tsky; comp1.temp, comp1.co2, comp1.illumination, comp1.irrigation; AirT, AirRH, Airppm
        训练Y为3维: AirT, AirRH, Airppm
        '''

        train_X = np.concatenate(X[:, :, :13], axis=0)
        train_Y = np.concatenate(Y[:, :, :3], axis=0)

        return train_X, train_Y

    def crop_front_x_y(self, X, Y):
        '''
        此函数用于产生第二阶段即作物模拟前段自身状态模拟的训练数据，
        数据粒度为小时级
        训练X为7维: AirT, AirRH, Airppm, PARsensor; LAI, PlantLoad, NetGrowth
        训练Y为3维: LAI, PlantLoad, NetGrowth
        '''

        train_X = np.concatenate(X[:, :, 10:17], axis=0)
        train_Y = np.concatenate(Y[:, :, 3:6], axis=0)

        return train_X, train_Y

    def crop_back_x_y(self, X, Y):
        '''
        此函数用于产生第三阶段即作物模拟后段果实成熟模拟的训练数据，
        数据粒度为小时级
        训练X为4维: LAI, PlantLoad, NetGrowth, FW
        训练Y为1维: FW
        '''
        DAY_IN_LIFE_CYCLE = 166
        day_index = [23 + i * 24 for i in range(DAY_IN_LIFE_CYCLE)]
        day_index_plus = [23 + (i + 1) * 24 for i in range(DAY_IN_LIFE_CYCLE)]
        train_X = np.concatenate(X[:, day_index, -4:], axis=0)
        train_Y = np.concatenate(X[:, day_index_plus, -1], axis=0)
        train_Y = train_Y.reshape(len(train_Y), -1)

        return train_X, train_Y


if __name__ == '__main__':
    pass
