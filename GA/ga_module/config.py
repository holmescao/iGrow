setting_train =\
    {
        "settings":
        {
            "parallel": "false",
        },
        "params_info":
        {
            "seed": 0,
            "NIND": 20,
            "MAXGEN": 50000,
            "XOVR": 0.3,
            "LINKDAY": 1,
        },
        "action_info":
        {
            'comp1.setpoints.temp.@heatingTemp': [24, 1, 24],
            'comp1.setpoints.CO2.@setpoint': [24, 50, 24],
            'comp1.illumination.lmp1.@turnOn': [24, 1, 24],
            'comp1.irrigation.@irriOn': [24, 1, 24],
        }
    }

setting_test =\
    {
        "params_info":
        {
            "seed": 31,
            "NIND": 6,
            "MAXGEN": 50000,
            "XOVR": 0.3,
            "LINKDAY": 1,
        },
        "action_info":
        {
            'comp1.setpoints.temp.@heatingTemp': [24, 1, 24],
            'comp1.setpoints.CO2.@setpoint': [24, 50, 24],
            'comp1.illumination.lmp1.@turnOn': [24, 1, 24],
            'comp1.irrigation.@irriOn': [24, 1, 24],
        }
    }
