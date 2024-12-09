import argparse
import numpy as np
# import matplotlib.pyplot as plt
from cityflow_env_wrapper import CityFlowEnvWrapper
import time
import os
import shutil
from configs import env_config, exp_config,region_config
import copy
from Pipeline_Aug import pipeline
import json
# tf.random.set_seed(0)
import shutil
import tensorflow as tf
tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if len(physical_devices)>1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
# np.random.se
# np.random.seed(0)
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--netname", type=str, default="Hangzhou")
    parser.add_argument("--netshape", type=str, default="4_4")
    parser.add_argument("--flow", type=str, default="real_5734")
    parser.add_argument("--agent", type=str, default="GAABDQ")
    parser.add_argument("--cell", type=int, default=5)
    parser.add_argument("--head",type=int, default=8)
    parser.add_argument("--region_type", type=str, default="ADJACENCY1")
    parser.add_argument("--folder", type=str, default="default")
    return parser.parse_args()



def init_exp(args):
    """
    Based on arguments and experiment configuration to initialise experiment
    Global working directory
    1.construct environment object
 
    2.agents object

    :param args:
    :return:
    """
    # retrieve arguments
    netname = args.netname
    netshape =  args.netshape
    flow =  args.flow
    agent_type = args.agent
    ENV_CONFIG = copy.deepcopy(env_config.ENV_CONFIG)
    EXP_CONFIG = copy.deepcopy(exp_config.EXP_CONFIG)
    EXP_CONFIG['AGENT_TYPE']=agent_type
    # Construct Cityflow Configuration File and ENV DICT

    roadnet_file = "roadnet_" + netshape + ".json"
    flow_file = netname + "_" + netshape + "_" + flow + ".json"
    net_path = os.path.join(netname, roadnet_file)
    flow_path = os.path.join(netname, flow_file)
    experiment_name = "{0}_{1}_{2}".format(netname, netshape, flow)
    experiment_date = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    ENV_CONFIG["PATH_TO_WORK_DIRECTORY"] = os.path.join(args.folder, experiment_name + "_" + experiment_date)

    if not os.path.exists(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"]):
        os.makedirs(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"])
    ENV_CONFIG["ROADNET_PATH"] = roadnet_file
    ENV_CONFIG["FLOW_PATH"] = flow_file
    ENV_CONFIG["CELL_COUNT"]=args.cell
    ENV_CONFIG["HEAD_NUM"]=args.head
    # copy net file and flow file to working directory
    shutil.copyfile(os.path.join("data/",net_path), os.path.join(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"],roadnet_file))
    shutil.copyfile(os.path.join("data/", flow_path), os.path.join(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"], flow_file))
    with open(os.path.join(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"], "env_config.json"), "w") as f:
        json.dump(ENV_CONFIG, f)

    env = CityFlowEnvWrapper(ENV_CONFIG)

    # Construct Agent
    # we need state dim , action dim, subaction num
    itsx_state_dim=24
    if ENV_CONFIG["ACTION_TYPE"]=="CHOOSE PHASE":
        itsx_action_dim=ENV_CONFIG["PHASECONFIG"]
    elif ENV_CONFIG["ACTION_TYPE"]=="SWTICH":
        itsx_action_dim=2
    else:
        raise Exception("Unknow phase control")
    if EXP_CONFIG["REGIONAL"]:

        region_obs_assignment_key=netshape+'_'+EXP_CONFIG["REGION_OB_RANGE"]
        temp_assign=copy.deepcopy(region_config.REGION_CONFIG[region_obs_assignment_key])
        itsx_assignment_obs=[]
        for assignment in temp_assign:
            itsx_assignment_obs.append([itsx_id for itsx_id in assignment if itsx_id != "dummy"])

        region_ex_assignment_key=netshape+'_'+EXP_CONFIG["REGION_EX_RANGE"]
        temp_assign= copy.deepcopy(region_config.REGION_CONFIG[region_ex_assignment_key])
        itsx_assignment_ex=[]
        for assignment in temp_assign:
            itsx_assignment_ex.append([itsx_id for itsx_id in assignment if itsx_id != "dummy"])

        itsx_assignment={
           "REGION_OB_RANGE":itsx_assignment_obs,
           "REGION_EX_RANGE":itsx_assignment_ex
        }

    else:
        itsx_assignment_ex=[[itsx_id] for itsx_id in env.intersection_ids]
        itsx_assignment_obs = [[itsx_id] for itsx_id in env.intersection_ids]
        itsx_assignment = {
            "REGION_OB_RANGE": itsx_assignment_obs,
            "REGION_EX_RANGE": itsx_assignment_ex
        }

    EXP_CONFIG["AGETN_NUM"]=len(itsx_assignment_ex)
    agent_config={
        "ITSX_STATE_DIM":itsx_state_dim,
        "ITSX_OBS_NUM":len(itsx_assignment_obs[0]),
        "ACTION_DIM":len(itsx_assignment_ex[0]),
        "ITSX_ACTION_DIM":itsx_action_dim,
        "PATH_TO_WORK_DIRECTORY":ENV_CONFIG["PATH_TO_WORK_DIRECTORY"],
        "ITSX_NUM": len(env.intersection_ids),
        "ITSX_ASSIGNMENT":itsx_assignment["REGION_EX_RANGE"],
        "CELL_COUNT": args.cell,
        "HEAD_NUM":args.head
    }

    EXP_CONFIG.update(agent_config)

    agent=EXP_CONFIG["AGENT_CLASS_DICT"][agent_type](0,agent_config)

    print(EXP_CONFIG)
    print(ENV_CONFIG)
    return env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG

def run_pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG):
    logs=pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG)
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_intersection_reward.npy'),logs['reward_log'])
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_throughput.npy'), logs['throughput'])
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_average_travel_time.npy'), logs['travel_time'])


# if __name__ == "__main__":
#     for i in range(8,12,3):
#         args = parse_args()
#         args.cell=i
#         env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG=init_exp(args)
#         run_pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG)
#     for name in dataset_list:
    # main(name)
# if __name__ == "__main__":
#     args = parse_args()
#     env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG=init_exp(args)
#     run_pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG)
#     # for name in dataset_list:
#     # main(name)

# if __name__ == "__main__":
#
#     netshape_list=["4_4","4_4","4_4","16_3"]
#     netname_list=["Hangzhou","Hangzhou","Syn","Manhattan"]
#     flow_list=["real","real_5734","gaussian_500_1h","real"]
#     for i in range (4):
#         args = parse_args()
#         args.netname = netname_list[i]
#         args.netshape = netshape_list[i]
#         args.flow = flow_list[i]
#         env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG=init_exp(args)
#         run_pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG)
#
# if __name__ == "__main__":
#
#     netshape_list=["4_4","4_4","4_4","16_3"]
#     netname_list=["Hangzhou","Hangzhou","Syn","Manhattan"]
#     flow_list=["real","real_5734","gaussian_500_1h","real"]
#     for i in range (1,4):
#         args = parse_args()
#         args.netname = netname_list[i]
#         args.netshape = netshape_list[i]
#         args.flow = flow_list[i]
#         for headnum in range(5,10):
#             if headnum==8:
#                 continue
#             if i==1 and (headnum==5 or headnum== 6):
#                 continue
#             args.head=headnum
#             env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG=init_exp(args)
#             run_pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG)

# if __name__ == "__main__":
#
#     netshape_list=["4_4","4_4","4_4","16_3"]
#     netname_list=["Hangzhou","Hangzhou","Syn","Manhattan"]
#     flow_list=["real","real_5734","gaussian_500_1h","real"]
#     for i in range (3):
#         args = parse_args()
#         args.netname = netname_list[i]
#         args.netshape = netshape_list[i]
#         args.flow = flow_list[i]
#         for cell_c in range(1,5):
#             args.cell=cell_c
#             env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG=init_exp(args)
#             run_pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG)
#     # for name in dataset_list:
#     # main(name)
if __name__ == "__main__":

    netshape_list=["4_4","4_4","4_4","16_3"]
    netname_list=["Hangzhou","Hangzhou","Syn","Manhattan"]
    flow_list=["real","real_5734","gaussian_500_1h","real"]
    for i in range (3,4):
        args = parse_args()
        args.netname = netname_list[i]
        args.netshape = netshape_list[i]
        args.flow = flow_list[i]
        for cell_c in range(1,3):
            args.cell=cell_c
            args.folder="records/GAABDQAugCell/cell_"+str(args.cell)
            env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG=init_exp(args)
            run_pipeline(env,agent,itsx_assignment,EXP_CONFIG,ENV_CONFIG)