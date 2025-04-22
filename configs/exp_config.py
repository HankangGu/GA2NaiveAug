from agentpool.AdaptiveBDQ_agent import AdaptiveBrachingDQ_agent
from agentpool.GAABDQ import GAABDQ

EXP_CONFIG = {
    "EPISODE": 2000,

    "TRAINING_PARADIM": "DECENTRAL",  # DECENTRAL",#"CLDE",  # CLDE: centralised learning but decentralised execution

    "REGIONAL": True,
    "REGION_OB_RANGE": "ADJACENCY1",
    "REGION_EX_RANGE": "ADJACENCY1",

    "AGENT_CLASS_DICT": {
        "ABDQ": AdaptiveBrachingDQ_agent,
        "GAABDQ":GAABDQ,

    },

    "LEARNING_INTERVAL": 5,
}
