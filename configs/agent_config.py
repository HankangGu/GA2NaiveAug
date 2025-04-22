# Basic agent config
AGENT_CONFIG = {
    "BATCH_SIZE": 32,##32,
    "MEMORY_SIZE": 200000,
    "MEMORY_TYPE": "RANDOM",#"PRIORITIZED",#"RANDOM",
    "MAX_EPSILON": 1,
    "MIN_EPSILON": 0.001,
    "DECAY_STEPS": 20000,#20000
    "NET_REPLACE_TYPE": "SOFT",
    "REPLACE_INTERVAL": 200,#,#200,
    "GAMMA": 0.9,

}

DQN_AGENT_CONFIG={
    "LEARNING_RATE": 0.0001,
"TAU": 0.001,
}
BDQ_AGENT_CONFIG = {
    "TD_OPERATOR": "MEAN",#"MEAN",
    "LEARNING_RATE": 0.0001,
    "TAU": 0.001,
}
GABDQ_AGENT_CONFIG = {
    "TD_OPERATOR": "MEAN",  # "MEAN",
    "LEARNING_RATE": 0.0001,
    "TAU": 0.001,
    "HEAD_NUM":8
}



if __name__=='__main__':
    print()