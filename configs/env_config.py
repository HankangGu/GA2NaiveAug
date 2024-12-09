
ENV_CONFIG={
    "SIM_TIMESPAN":4000,
    "ACTION_INTERVAL":20,
    "YELLOW_DURATION":3,
    "ACTION_TYPE":"CHOOSE PHASE",
    "INTERVAL":1.0,
    "SAVEREPLAY":False,
    "LOG": False,
    "RLTRAFFICLIGHT":True,
    "PHASECONFIG":4, #8
    "STATE_RETURN_TYPE": "RAW",
    "STATE":[
        "waiting_queue_length",
        # "waiting_queue_length_neibor_lane",
        "wave",
        "current_phase",


        # "waiting_queue_length_seg",
        # "",
    ],
    "REWARD":{
        "pressure":0,
        "sum_wait":-1,
        "waiting_len_difference":0,
        # "total_delay":-1,
        # "delay_average":0
    },
    "CELL_COUNT":5
}
