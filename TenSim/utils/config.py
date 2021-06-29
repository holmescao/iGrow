# -*- coding: utf-8 -*-
# @Author : Richardan@tencent
# @Time : 2020/9/22 17:26
# from iGrowOS.common.utils import get_path

'''
环境配置，目前主要包括WUR_Tomato_v0_full配置和Jinyu_Tomato_v0配置
WUR_Tomato_v0_full下包含了所有的模拟器可以操控的变量，但也需要在使用时针对这些变量进行筛选
'''

WUR_Tomato_v0_full = {
    'name': 'WUR-Tomato-v0-full',
    'desc': 'the full simulation of WUR tomato simulator',
    'step': 'day',
    'url': 'https://www.digigreenhouse.wur.nl/TenCent3/',
    'key': 'Axiany_Simulator2_6c66be20-4abc-451',
    'proxy_flag': True,
    'save_path': None,
    'start_date': '2019-12-16',
    'end_date': '2020-5-30',
    'observation': {
        "Iglob": {
            "min": 0.0,
            "max": 800.0,
            "dim": 24,
            "desc": "outside solar radiation",
        },
        "TOut": {
            "min": -10.0,
            "max": 30.0,
            "dim": 24,
            "desc": "outside temperature"
        },
        "RHout": {
            "min": 40.0,
            "max": 100.0,
            "dim": 24,
            "desc": "outside humidity"
        },
        "Windsp": {
            "min": 0.0,
            "max": 25.0,
            "dim": 24,
            "desc": "wind speed"
        },
        "AirT": {
            "min": 10.0,
            "max": 35.0,
            "dim": 24,
            "desc": "greenhouse air temperature"
        },
        "AirRH": {
            "min": 40.0,
            "max": 100.0,
            "dim": 24,
            "desc": "greenhouse air humidty "
        },
        "Airppm": {
            "min": 400.0,
            "max": 1000.0,
            "dim": 24,
            "desc": "greenhouse air CO2 concentration"
        },
        "PARsensor": {
            "min": 0.0,
            "max": 300.0,
            "dim": 24,
            "desc": "light intensity just above crop"
        },
        "TPipe1": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Average heating pipe1 temperature"
        },
        "TPipe2": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Average heating pipe2 temperature"
        },
        "TSupPipe1": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Supply side heating pipe1 temperature"
        },
        "TSupPipe2": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Supply side heating pipe2 temperature"
        },
        "PConPipe1": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Heating power to the heating system"
        },
        "PConPipe2": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Heating power to the heating system"
        },
        "WinLee": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Lee side vent opening"
        },
        "WinWnd": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "Windward side vent opening"
        },
        "SpHeat": {
            "min": 0.0,
            "max": 60.0,
            "dim": 24,
            "desc": "Setpoint for heating"
        },
        "SpVent": {
            "min": 0.0,
            "max": 60.0,
            "dim": 24,
            "desc": "Ventilation line"
        },
        "Scr1": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": "Position of the first screen"
        },
        "Scr2": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": "Position of the second screen"
        },
        "Lmp1": {
            "min": 0.0,
            "max": 300.0,
            "dim": 24,
            "desc": "Lamps on or off"
        },
        "McConAir": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "CO2 supply rate"
        },
        "SlabEC": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": "EC-value of the slab"
        },
        "Irrigation": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": "Cumulative amount of irrigation per day"
        },
        "Drain": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": "Cumulative amount of drain per day"
        },
        "LightSum": {
            "min": 0.0,
            "max": 30.0,
            "dim": 24,
            "desc": "Cumulative amount of PAR light per day"
        },
        "LAI": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": "Leaf area index"
        },
        "CumFruitsDW": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": "cumulative harvesr in terms of fruit dry weight"
        },
        "CumFruitsFW": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": "cumulative harvesr in terms of fruit fresh weight"
        },
        "CumFruitsCount": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        "Brix": {
            "min": 0.0,
            "max": 30.0,
            "dim": 24,
            "desc": ""
        },
        "Shoots": {
            "min": 0.0,
            "max": 30.0,
            "dim": 24,
            "desc": ""
        },
        "PlantLoad": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": "current number of growing fruits"
        },
        "PeakHour": {
            "min": 0.0,
            "max": 24.0,
            "dim": 24,
            "desc": ""
        },
        "NetGrowth": {
            "min": -10.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        "PowerFac": {
            "min": 0.0,
            "max": 11.0,
            "dim": 24,
            "desc": ""
        },
        "Idir": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "Idiff": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        }
    },
    'action': {
        "common.CO2dosing.@pureCO2cap": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.heatingpipes.pipe1.@minTemp": {
            "min": 0.0,
            "max": 60.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.heatingpipes.pipe1.@maxTemp": {
            "min": 0.0,
            "max": 60.0,
            "dim": 24,
            "desc": ""
        },
        # "comp1.heatingpipes.pipe1.@radiationInfluence": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        "comp1.screens.scr1.@enabled": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": ""
        },
        # "comp1.screens.scr1.@material": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        # "comp1.screens.scr1.@closeBelow": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        "comp1.screens.scr1.@ToutMax": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.screens.scr1.@lightPollutionPrevention": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.screens.scr2.@enabled": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": ""
        },
        # "comp1.screens.scr2.@material": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        # "comp1.screens.scr2.@closeBelow": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        "comp1.screens.scr2.@closeAbove": {
            "min": 1000.0,
            "max": 1200.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.screens.scr2.@ToutMax": {
            "min": 1.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.screens.scr2.@lightPollutionPrevention": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.illumination.lmp1.@intensity": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.illumination.lmp1.@maxIglob": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.illumination.lmp1.@hoursLight": {
            "min": 0.0,
            "max": 24.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.illumination.lmp1.@endTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.illumination.lmp1.@maxPARsum": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.temp.@heatingTemp": {
            "min": 0.0,
            "max": 60.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.temp.@ventOffset": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        # "comp1.setpoints.temp.@PbandVent": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        # "comp1.setpoints.temp.@radiationInfluence": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        "comp1.setpoints.humidity.@setpoint": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.humidity.@PbandVent": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.CO2.@setpoint": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.CO2.@doseCapacity": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.CO2.@setpIfLamps": {
            "min": 0.0,
            "max": 10000.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.ventilation.@winLeeMin": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.ventilation.@winLeeMax": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.ventilation.@winWndMin": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.ventilation.@winWndMax": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.setpoints.ventilation.@startWnd": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.irrigation.@enabled": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.irrigation.@shotSize": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.irrigation.@maxPauseTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.irrigation.@molesPerShot": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.irrigation.@startTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.irrigation.@stopTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.irrigation.@EC": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.airconditioning.fogging.@enabled": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.airconditioning.fogging.@capacity": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.airconditioning.fogging.@minTemp": {
            "min": 0.0,
            "max": 60.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.airconditioning.fogging.@Pband": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        "comp1.airconditioning.fogging.@RHstart": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        # "crp_Axiany.general.@lowFruitWpunishment": {
        #     "min": 0.0,
        #     "max": 100.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        "crp_Axiany.general.@productPrice": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        },
        "crp_Axiany.Intkam.management.@stemDensity": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        # "crp_Axiany.Intkam.management.@leafPickingStrategy": {
        #     "min": 0.0,
        #     "max": 10.0,
        #     "dim": 24,
        #     "desc": ""
        # },
        "crp_Axiany.Intkam.management.@dayTopping": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 24,
            "desc": ""
        },
        "crp_Axiany.Intkam.LAI.@targetLAI": {
            "min": 0.0,
            "max": 10.0,
            "dim": 24,
            "desc": ""
        },
        "crp_Axiany.Intkam.tomato.growth.@FruitNrPerTruss": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": ""
        }
    },
    'reward': 'CumFruitsFW'
}

Jinyu_Tomato_v0 = {
    'name': 'jinyu-tomato-v0',
    'desc': 'self-trained simulator model',
    'step': 'day',
    'start_date': '2019-12-16',
    'end_date': '2020-5-24',
    'model_dir': None,
    'observation': {
        'Cropdays': {
            'min': 0,
            'max': 365,
            'dim': 1,
            'desc': 'planting days'
        },
        "comp1.setpoints.temp.@heatingTemp": {
            "min": 13.0,
            "max": 32.0,
            "dim": 24,
            "desc": "temperature setpoint"
        },
        "comp1.setpoints.CO2.@setpoint": {
            "min": 400.0,
            "max": 1000.0,
            "dim": 24,
            "desc": "CO2 setpoint"
        },
        "comp1.illumination.lmp1.@hoursLight": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "light on duration"
        },
        "comp1.illumination.lmp1.@endTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "turn off time"
        },
        "comp1.irrigation.@startTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "irrigation starttime"
        },
        "comp1.irrigation.@endTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "irrigation endtime"
        },
        "Iglob": {
            "min": 0.0,
            "max": 2000.0,
            "dim": 24,
            "desc": "outside solar radiation",
        },
        "TOut": {
            "min": -30.0,
            "max": 50.0,
            "dim": 24,
            "desc": "outside temperature"
        },
        "RHout": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "outside humidity"
        },
        "co2out": {
            "min": 400.0,
            "max": 1000.0,
            "dim": 24,
            "desc": "outside co2"
        },
        "Windsp": {
            "min": 0.0,
            "max": 25.0,
            "dim": 24,
            "desc": "wind speed"
        },
        "Tsky": {
            "min": -20.0,
            "max": 20.0,
            "dim": 24,
            "desc": "virtual sky tempurature"
        },
        "AirT": {
            "min": -30.0,
            "max": 100.0,
            "dim": 24,
            "desc": "greenhouse air temperature"
        },
        "AirRH": {
            "min": 0.0,
            "max": 100.0,
            "dim": 24,
            "desc": "greenhouse air humidty "
        },
        "Airppm": {
            "min": 400.0,
            "max": 1000.0,
            "dim": 24,
            "desc": "greenhouse air CO2 concentration"
        },
        "PARsensor": {
            "min": 0.0,
            "max": 2000.0,
            "dim": 24,
            "desc": "light intensity just above crop"
        },
        "Irrigation": {
            "min": 0.0,
            "max": 10.0,
            "dim": 1,
            "desc": "cumulative amount of irrigation per day"
        },
        "Drain": {
            "min": 0.0,
            "max": 10.0,
            "dim": 1,
            "desc": "cumulative amount of drain per day"
        },
        "LAI": {
            "min": 0.0,
            "max": 10.0,
            "dim": 1,
            "desc": "leaf area index"
        },
        "PlantLoad": {
            "min": 0.0,
            "max": 1000.0,
            "dim": 1,
            "desc": "current number of growing fruits"
        },
        "CumFruitsFW": {
            "min": 0.0,
            "max": 30.0,
            "dim": 1,
            "desc": "cumulative harvesr in terms of fruit fresh weight"
        },
        "CumFruitsDW": {
            "min": 0.0,
            "max": 30.0,
            "dim": 1,
            "desc": "cumulative harvesr in terms of fruit dry weight"
        }
    },
    'action': {
        "comp1.setpoints.temp.@heatingTemp": {
            "min": 13.0,
            "max": 32.0,
            "dim": 24,
            "desc": "temperature setpoint"
        },
        "comp1.setpoints.CO2.@setpoint": {
            "min": 400.0,
            "max": 1000.0,
            "dim": 24,
            "desc": "CO2 setpoint"
        },
        "comp1.illumination.lmp1.@hoursLight": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "light on duration"
        },
        "comp1.illumination.lmp1.@endTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "turn off time"
        },
        "comp1.illumination.lmp1.@turnOn": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": "turn on/off"
        },
        "comp1.irrigation.@startTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "irrigation starttime"
        },
        "comp1.irrigation.@endTime": {
            "min": 0.0,
            "max": 24.0,
            "dim": 1,
            "desc": "irrigation endtime"
        },
        "comp1.irrigation.@irriOn": {
            "min": 0.0,
            "max": 1.0,
            "dim": 24,
            "desc": "irrigation or not"
        }
    },
    'reward': 'CumFruitsFW'
}
