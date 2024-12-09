import numpy as np
import time
import os
import pickle
import networkx as nx
import tensorflow as tf
import itertools


def build_itsx_graph(itsx_ids):
    G = nx.Graph()
    for itsx1, itsx2 in itertools.combinations(itsx_ids, 2):
        _, itsx1_x, itsx1_y = itsx1.split('_')
        _, itsx2_x, itsx2_y = itsx2.split('_')
        if abs(int(itsx1_x) - int(itsx2_x)) + abs(int(itsx1_y) - int(itsx2_y)) <= 1:
            G.add_edge(itsx1, itsx2)
    return G


def build_support_info(itsx_assignment, movements):
    movement_graph = nx.Graph()
    in_lane_id_list = []  # store the order of lane id for generating movement adj matrix
    itsx_id_list = []  # store the order of itsx id for generating itsx matrix
    # build itsx-level state
    for assignment in itsx_assignment:
        for itsx_id in assignment:
            if itsx_id == "dummy":
                continue
            itsx_id_list.append(itsx_id)
            # build lane-level state and movement graph
            for in_lane_id, out_lane_id in movements[itsx_id].items():
                movement_graph.add_edge(in_lane_id,in_lane_id)
                movement_graph.add_edge(in_lane_id, out_lane_id)
                in_lane_id_list.append(in_lane_id)
    itsx_graph = build_itsx_graph(itsx_id_list)
    itsx_adj_matrix = nx.adjacency_matrix(itsx_graph, itsx_id_list).todense()
    lane_level_adj_matrix = nx.adjacency_matrix(movement_graph, in_lane_id_list).todense()

    return in_lane_id_list, lane_level_adj_matrix, itsx_id_list, itsx_adj_matrix


def assign_state(state, lane_order, itsx_order):
    def _concat_state(itsx_state):
        temp = []
        for key, value in itsx_state.items():
            temp.extend(value)
        return np.array(temp)

    itsx_level_state = []
    lane_level_state = []
    # build itsx-level state

    itsx_level_global_state = {}
    ll_global_state = {}
    for key, value in state.items():
        itsx_level_global_state[key] = {"waiting_queue_length": value["waiting_queue_length"],
                                        "wave": value["wave"]}
        ll_global_state.update(value["discretized_wave"])

    for itsx_id in itsx_order:
        current_state = itsx_level_global_state[itsx_id]
        itsx_level_state.append(_concat_state(current_state))
        # build lane-level state and movement graph
    for in_lane_id in lane_order:
        lane_level_state.append(ll_global_state[in_lane_id])

    return np.array(lane_level_state), np.array(itsx_level_state)


def assign_reward(raw_reward, itsx_assignment, itsx_order):
    """
    build regional reward from raw reward of intersections for each agent according to the intersection assignments
    :param raw_reward: a dictionary {key,value} key(String): intersection id,
                                               value(int): reward of that intersection
    :return:
    """
    itsx_reward = {}
    for assignment in itsx_assignment:
        if len(assignment) == 0:
            assignment = [assignment]
        agent_reward = 0  # reward of current agent

        for _, itsx in enumerate(assignment):

            agent_reward += raw_reward[itsx]
        itsx_reward.update(dict.fromkeys(assignment, agent_reward))
    ordered_itsx_reward = []
    for itsx in itsx_order:
        ordered_itsx_reward.append(itsx_reward[itsx])
    return ordered_itsx_reward


def convert_actions(actions_id, itsx_order):
    decoded_actions = {}
    for action, itsx in zip(actions_id, itsx_order):
        decoded_actions[itsx] = int(action) + 1
    return decoded_actions


def pipeline(env, agent, itsx_assignment, EXP_CONFIG, ENV_CONFIG):
    """
    Training pipeline that communicate agent with environment

    :param env:
    :param agents:
    :param itsx_assignment:
    :param EXP_CONFIG:
    :param ENV_CONFIG:
    :return:

    """

    def agent_learn():
        agent.learn()



    # print(env.get_movement_graph())
    movements = env.get_movements()
    lane_order, ll_adj, itsx_order, itsx_adj = build_support_info(itsx_assignment["REGION_EX_RANGE"], movements)
    agent.update_support_info(ll_adj, itsx_adj, itsx_order, itsx_assignment["REGION_EX_RANGE"])
    """----init global log-----"""
    global_step = 0  # global training step
    episode_intersection_level_rewards = []
    episode_throughput = []
    episode_travel_time = []
    """-------------"""
    for episode in range(EXP_CONFIG["EPISODE"]):
        state = env.reset(round=episode)
        ll_state, itsx_state = assign_state(state, lane_order, itsx_order)
        # print(np.sum(ll_adj, axis=1))
        """---init episode log---"""
        episode_start_time = time.time()
        step_itsx_reward = []
        episode_reward = []
        signal_phase_his = []
        # action_time=0

        """-------"""
        for step in range(int(ENV_CONFIG["SIM_TIMESPAN"] / ENV_CONFIG["ACTION_INTERVAL"])):
            global_step += 1

            # agents make decisions
            # start_time = time.time()
            actions_id = agent.choose_action(ll_state, ll_adj, itsx_state, itsx_adj)
            # end_time = time.time()
            # action_time += end_time - start_time
            joint_actions = convert_actions(actions_id, itsx_order)
            # print(joint_actions)

            next_states, itsx_rewards, done, log_metric = env.step(joint_actions)  # simulate

            # assign observations and reward
            next_ll_state, next_itsx_state = assign_state(next_states, lane_order, itsx_order)
            rewards = assign_reward(itsx_rewards, itsx_assignment["REGION_EX_RANGE"], itsx_order)

            if not done:
                # only store experience and learn when not finished

                agent.store_transition(ll_state, itsx_state, actions_id, rewards, next_ll_state, next_itsx_state)
                if global_step % EXP_CONFIG["LEARNING_INTERVAL"] == 0:
                    agent_learn()
            ll_state = next_ll_state
            itsx_state = next_itsx_state
            """----update step log---"""
            step_itsx_reward.append([value for _, value in itsx_rewards.items()])
            episode_reward.append(np.average(rewards))
            """-----------------"""
        # print("action time,", action_time/(int(ENV_CONFIG["SIM_TIMESPAN"] / ENV_CONFIG["ACTION_INTERVAL"])))
        # break
        """----update episode log------"""
        episode_throughput.append(env.get_throughput())
        episode_travel_time.append(env.get_average_travel_time())
        episode_intersection_level_rewards.append(np.array(step_itsx_reward))
        """----------------------------"""
        # if episode>1:
        #     save_agent()

        log_msg = {
            "episode": episode,
            "time cost": time.time() - episode_start_time,
            "episode reward": np.sum(episode_reward),
            "average travel time": env.get_average_travel_time(),
            "epsilon": agent.epsilon

        }
        print(log_msg)
    sim_log = {'reward_log': episode_intersection_level_rewards,
               'throughput': episode_throughput,
               'travel_time': episode_travel_time
               }
    # pickle.dump(episode_intersection_level_rewards,open(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_intersection_reward.npy'),'wb'))
    # pickle.dump(episode_throughput,open(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_throughput.npy'),'wb'))
    # pickle.dump(episode_travel_time,open(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_average_travel_time.npy'),'wb'))
    # np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY']),episode_intersection_level_rewards)
    # np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_throughput.npy'),episode_throughput)
    # np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_average_travel_time.npy'),episode_travel_time)
    return sim_log
