import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box
import gym

gym.logger.set_level(40)
torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=120):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


class Economic:
    def __init__(self, dayAction, dayState, dayReward, dayCrop):
        self.dayAction = dayAction  # [temp, co2, illu, irri]
        self.dayState = dayState  # [AirT, AirRH, Airppm]
        self.dayReward = dayReward  # [FW]
        self.dayCrop = dayCrop  # [stem]

    @property
    def cal_economic(self):
        # 累计经济
        gains = self.cal_gains
        elecCost = self.elec_cost
        co2Cost = self.co2_cost
        heatCost = self.heat_cost
        laborCost = self.labor_cost

        variableCosts = elecCost + co2Cost + heatCost + laborCost
        balance = gains - variableCosts

        economic = {'balance': balance,
                    'gains': gains,
                    'variableCosts': variableCosts,
                    'elecCost': elecCost,
                    'co2Cost': co2Cost,
                    'heatCost': heatCost,
                    'laborCost': laborCost}

        return economic

    @property
    def cal_gains(self):
        # price = 3.5
        price = 3.185
        # price = 3.1
        # price = 2.94
        return self.dayReward[0] * price

    @property
    def elec_cost(self):
        lmp_use = self.dayAction[:, 2]
        days = len(lmp_use) // 24

        power = 185 / 2.1

        price = np.array(([0.04] * 7 + [0.08] * 16 + [0.04] * 1) * days)
        cost = np.sum(np.array(lmp_use) * price * power / 1000)

        return cost

    @property
    def co2_cost(self):
        CO2_setpoint = self.dayAction[:, 1]
        Airppm = self.dayState[:, 2]

        McConAir_max = 4e-6
        co2_use = CO2_setpoint - Airppm
        co2_use[co2_use > 0] = McConAir_max
        co2_use[co2_use <= 0] = 0

        price1, price2 = 0.08, 0.2
        kgCO2 = sum(co2_use) * 3600
        firstTranche = min(kgCO2, 12)
        secondTranche = kgCO2 - firstTranche
        cost = firstTranche * price1 + secondTranche * price2

        return cost

    @property
    def heat_cost(self):
        temp_setpoint = self.dayAction[:, 0]
        AirT = self.dayState[:, 0]

        PConPipe1_max = np.full((24), 60)
        heat_use = (PConPipe1_max - AirT) * 2.1

        heat = temp_setpoint - AirT
        heat_use[heat <= 0] = 0

        price = 0.03
        cost = sum(heat_use) * price / 1000

        return cost

    @property
    def labor_cost(self):
        stems = self.dayCrop[0]
        price = 0.0085
        cost = stems * price

        return cost


class PredictModel(gym.Env):

    def __init__(self,
                 model1_dir,
                 model2_dir,
                 model3_dir,
                 scaler1_x,
                 scaler1_y,
                 scaler2_x,
                 scaler2_y,
                 scaler3_x,
                 scaler3_y,
                 linreg_dir,
                 weather_dir):
        self.net1 = Net(13, 3, 300)
        self.net1.load_state_dict(torch.load(
            model1_dir, map_location=torch.device('cpu')))
        self.net2 = Net(7, 3, 300)
        self.net2.load_state_dict(torch.load(
            model2_dir, map_location=torch.device('cpu')))
        self.net3 = Net(4, 1, 600)
        self.net3.load_state_dict(torch.load(
            model3_dir, map_location=torch.device('cpu')))
        self.net1.eval()
        self.net2.eval()
        self.net3.eval()

        self.scaler1_x = pickle.load(open(scaler1_x, 'rb'))
        self.scaler1_y = pickle.load(open(scaler1_y, 'rb'))
        self.scaler2_x = pickle.load(open(scaler2_x, 'rb'))
        self.scaler2_y = pickle.load(open(scaler2_y, 'rb'))
        self.scaler3_x = pickle.load(open(scaler3_x, 'rb'))
        self.scaler3_y = pickle.load(open(scaler3_y, 'rb'))

        self.linreg = pickle.load(open(linreg_dir, 'rb'))

        self.full_weather = np.load(weather_dir)

        # 重置环境
        self.reset()

        self._max_episode_steps = 166

    @property
    def observation_space(self):
        # AirT * 24, AirRH * 24, Airppm * 24; LAI, PlantLoad, NetGrowth; FW
        low = np.concatenate(([10 for _ in range(24)], [0 for _ in range(24)], [
                             300 for _ in range(24)], [0, 0, 0, 0]))
        high = np.concatenate(
            ([35 for _ in range(24)], [100 for _ in range(24)], [1000 for _ in range(24)], [6, 1000, 0.002, 20]))
        return Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self):
        # temp, co2, illumination, irrigation
        low = np.concatenate(
            ([13 for _ in range(24)], [400 for _ in range(24)], [0 for _ in range(24)], [0 for _ in range(24)]))
        high = np.concatenate(
            ([33 for _ in range(24)], [1000 for _ in range(24)], [1 for _ in range(24)], [1 for _ in range(24)]))
        return Box(low=low, high=high, dtype=np.float32)

    def get_outside_weather(self, day_index):
        if day_index < 160:
            return self.full_weather[day_index * 24:(day_index + 1) * 24]
        else:
            mv_idx = day_index - 160
            return self.full_weather[(day_index - mv_idx) * 24:(day_index + 1-mv_idx) * 24]

    def calculate_reward(self, dayAction, dayState, dayReward, dayCrop):
        economic = Economic(dayAction=dayAction, dayState=dayState,
                            dayReward=dayReward, dayCrop=dayCrop).cal_economic
        reward = economic['balance']

        return reward, economic

    def update_observation(self, day_inside_weather):
        # 更新状态值
        day_inside_weather = day_inside_weather.T.reshape(1, -1)
        self.observation = np.hstack(
            (day_inside_weather[0], self.crop_state, self.fw))
        # 是否进行截断
        # self.observation = np.clip(self.observation,self.observation_space.low,self.observation_space.high)

    def step(self, action):
        assert len(action) == 96, 'wrong input control dimension'
        # assert len(action) == 24, 'wrong input control dimension'

        # 更新动作值
        self.action = action
        # print(self.action)

        action = action.reshape((4, 24)).T
        # action = action.reshape((4, 6)).T
        # action[2,0] = 32
        # action = action.repeat(4, axis=0)

        day_outside_weather = self.get_outside_weather(self.day_index)

        day_inside_weather = np.zeros((24, 3))  # 保存一天内的内部天气状况

        for i in range(24):
            # each hour
            self.outside_weather = day_outside_weather[i]
            # Igolb, Tout, RHout, Co2out, Windsp, Tsky
            cur_outside_weather = self.outside_weather
            cur_control = action[i]  # temp, co2, illu, irri
            cur_inside_weather = self.inside_weather  # AirT, AirRH, Airppm
            day_inside_weather[i, :] = cur_inside_weather
            input1 = np.hstack(
                (cur_outside_weather, cur_control, cur_inside_weather))
            input1 = input1.reshape(1, -1)
            # input1 = input1.repeat(5,axis=0)
            input1_normal = self.scaler1_x.transform(input1)
            input1_normal = torch.tensor(input1_normal, dtype=torch.float)
            output1_normal = self.net1(input1_normal).detach().numpy()
            # output1_normal = output1_normal.reshape(1,-1)
            output1 = self.scaler1_y.inverse_transform(output1_normal)[0]

            self.inside_weather = output1
            # PARsensor calculation
            PARsensor = self.linreg.predict(input1[0, [0, 8]].reshape(1, -1))
            PARsensor = PARsensor if PARsensor > 50.0 else 0.0
            PARsensor = np.array(PARsensor)
            # PARsensor = PARsensor.reshape(1,-1)

            input2 = np.hstack(
                (self.inside_weather, PARsensor, self.crop_state))
            input2 = input2.reshape(1, -1)
            input2_normal = self.scaler2_x.transform(input2)
            input2_normal = torch.tensor(input2_normal, dtype=torch.float)
            output2_normal = self.net2(input2_normal).detach().numpy()
            output2 = self.scaler2_y.inverse_transform(output2_normal)[0]

            # output2[-1] = np.maximum(self.crop_state[-1], output2[-1])

            self.crop_state = output2  # LAI, PlantLoad, NetGrowth
            # print(self.inside_weather)

        input3 = np.concatenate((self.crop_state, self.fw))
        input3 = input3.reshape(1, -1)
        input3_normal = self.scaler3_x.transform(input3)
        input3_normal = torch.tensor(input3_normal, dtype=torch.float)
        output3_normal = self.net3(input3_normal).detach().numpy()
        output3 = self.scaler3_y.inverse_transform(output3_normal)[0]

        # 前一天产量
        cur_fw = self.fw
        # 当前预测产量
        output3 = output3 if output3 > 0.1 else np.array([0])
        output3 = np.maximum(output3, self.fw)
        self.fw = output3

        # 增量
        day_fw = (self.fw - cur_fw) + self.store_fw
        harvest = 0.6
        if day_fw < harvest:
            self.store_fw += self.fw - cur_fw  # 保存未成熟的
            day_fw = np.zeros(1)  # 成熟的置0
        else:
            # 存量
            self.store_fw = np.zeros(1)

        dayCrop = self.periodCrop[self.day_index:self.day_index+1]
        reward, economic = self.calculate_reward(
            dayAction=action, dayState=day_inside_weather, dayReward=day_fw, dayCrop=dayCrop)
        self.update_observation(day_inside_weather)

        self.day_index += 1

        done = self.day_index >= self._max_episode_steps

        return self.observation, reward, done, economic

    def reset(self, periodCrop=np.full(166, 3.9)):
        # 初始化动作值
        # temp_list = [28 for _ in range(24)]
        # co2_list = [800 for _ in range(24)]
        # illu_list = [1 for _ in range(24)]
        # irri_list = [0 for _ in range(24)]
        # self.action = np.hstack((temp_list, co2_list, illu_list, irri_list))
        temp_list = [23 for _ in range(6)]
        co2_list = [598 for _ in range(6)]
        # temp_list = [18,20,22,28,30,32]
        # co2_list = [400,500,600,700,800,900]
        illu_list = [0 for _ in range(6)]
        irri_list = [0 for _ in range(6)]
        self.action = np.hstack((temp_list, co2_list, illu_list, irri_list))

        # 初始化状态值
        # 初始化外部天气
        self.outside_weather = self.get_outside_weather(0)[0]
        # 初始化内部天气
        self.inside_weather = np.array([17.27, 61.83, 737.31])
        # self.day_inside_weather = np.concatenate(
        #     ([22 for _ in range(24)], [60 for _ in range(24)], [700 for _ in range(24)]))
        self.day_inside_weather = np.concatenate(
            (np.random.randint(20, 24, 24), np.random.randint(50, 80, 24), np.random.randint(600, 800, 24)))
        # 初始化作物状态
        self.crop_state = np.array([0.2, 0, 0])
        # 初始化FW
        self.fw = np.zeros(1)

        self.max_plantlaod = np.zeros(1)

        self.periodCrop = periodCrop
        # 存量
        self.store_fw = np.zeros(1)

        self.observation = np.hstack(
            (self.day_inside_weather, self.crop_state, self.fw))

        # 初始化种植日期索引
        self.day_index = 0

        # return self.action
        return self.observation
