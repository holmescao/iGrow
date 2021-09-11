config = {
    'name': 'igrow-tomato-v1',
    'desc': 'self-trained simulator model',
    'step': 'day',
    'start_date': '2019-12-16',
    'end_date': '2020-5-29',
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
