
import tensorflow as tf
from tensorflow.keras import layers
import copy
import numpy as np
import os
from configs import agent_config
import copy
import time
import json

class GAABDQ:
    def __init__(self,
                 agent_id,
                 env_config
                 ):

        self.agent_id = agent_id
        self.subaction_num = env_config["ITSX_ACTION_DIM"]
        self.state_dim = env_config["ITSX_STATE_DIM"]
        self.itsx_num = env_config["ITSX_NUM"]
        self.cell_count=env_config["CELL_COUNT"]
        self.head_num=env_config["HEAD_NUM"]
        AGENT_CONFIG = copy.deepcopy(agent_config.AGENT_CONFIG)
        AGENT_CONFIG.update(agent_config.GABDQ_AGENT_CONFIG)
        AGENT_CONFIG["HEAD_NUM"]=self.head_num
        print(AGENT_CONFIG)
        with open(os.path.join(env_config["PATH_TO_WORK_DIRECTORY"],"agent_config.json"),"w") as f:
            json.dump(AGENT_CONFIG,f)

        # build model
        self.eval_model = build_network(self.itsx_num, self.state_dim, self.subaction_num,env_config["ITSX_ASSIGNMENT"],cell_length=self.cell_count,head_num=AGENT_CONFIG["HEAD_NUM"])
        # print(self.eval_model.summary())
        self.target_model = build_network(self.itsx_num, self.state_dim, self.subaction_num,env_config["ITSX_ASSIGNMENT"],cell_length=self.cell_count,head_num=AGENT_CONFIG["HEAD_NUM"])
        self.target_model.set_weights(self.eval_model.get_weights())

        self.td_operator_type = AGENT_CONFIG["TD_OPERATOR"]
        # training hyper parameter

        self.max_epsilon = AGENT_CONFIG["MAX_EPSILON"]
        self.epsilon = self.max_epsilon
        self.min_epsilon = AGENT_CONFIG["MIN_EPSILON"]
        self.decay_steps = AGENT_CONFIG["DECAY_STEPS"]
        self.gamma = AGENT_CONFIG["GAMMA"]
        self.learn_count = 0
        self.replace_target_iter = AGENT_CONFIG["REPLACE_INTERVAL"]
        self.replace_count = 0

        # memory replay

        self.memory_counter = 0
        self.memory_size = AGENT_CONFIG["MEMORY_SIZE"]
        self.batch_size = AGENT_CONFIG["BATCH_SIZE"]


        self.ll_state_memory = np.zeros((self.memory_size, self.itsx_num * 12, self.cell_count))
        self.itsx_state_memory = np.zeros((self.memory_size, self.itsx_num, 24))
        self.action_memory = np.zeros((self.memory_size, self.itsx_num))
        self.reward_memory = np.zeros((self.memory_size, self.itsx_num))
        self.next_ll_state_memory = np.zeros((self.memory_size, self.itsx_num * 12,self.cell_count))
        self.next_itsx_state_memory = np.zeros((self.memory_size, self.itsx_num, 24))

        self.replace_mode = AGENT_CONFIG["NET_REPLACE_TYPE"]
        if self.replace_mode == 'HARD':
            self.tau = 1
        else:
            self.tau = AGENT_CONFIG["TAU"]
        self.loss_his = []
        #
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=AGENT_CONFIG["LEARNING_RATE"])
        # print()

        self.work_folder = env_config['PATH_TO_WORK_DIRECTORY']

    def choose_action(self, ll_state, ll_adj, itsx_state, itsx_adj):
        """
        in dynamic DBQ, some branches are not activated.
        The idle_id store which branch is not activated
        for idle branches, the action is replaced by -1.
        :param obs: observation vector
        :param idle_id: the id of idle branches
        :param pack_value: whether return the q-value of each branch
        :return:
        """
        # print()
        ll_state = ll_state[np.newaxis, :]
        itsx_state = itsx_state[np.newaxis, :]
        ll_adj = ll_adj[np.newaxis, :]
        itsx_adj = itsx_adj[np.newaxis, :]
        action_branch_value = self.eval_model([ll_state, ll_adj, itsx_state, itsx_adj]).numpy()
        if np.random.random() > self.epsilon:
            # greedy choice
            joint_action = np.squeeze(np.argmax(action_branch_value, axis=2))

        else:
            # random
            joint_action = np.random.randint(0, 4, size=(self.itsx_num,))

        # set idle branch
        return joint_action

    def learn(self):
        # print("learn")
        available_count = min(self.memory_size, self.memory_counter)
        self.learn_count += 1
        if available_count > 10000:  # 10000
            self.replace_count += 1

            batch_indices = np.random.choice(available_count, self.batch_size)
            ll_state = tf.convert_to_tensor(self.ll_state_memory[batch_indices])
            itsx_state = tf.convert_to_tensor(self.itsx_state_memory[batch_indices])
            a = tf.convert_to_tensor(self.action_memory[batch_indices])
            r = tf.convert_to_tensor(self.reward_memory[batch_indices])
            r = tf.cast(r, dtype=tf.float32)
            next_ll_state = tf.convert_to_tensor(self.next_ll_state_memory[batch_indices])
            next_itsx_state = tf.convert_to_tensor(self.next_itsx_state_memory[batch_indices])
            # update_start=time.time()
            self.update_gradient(ll_state, itsx_state, a, r, next_ll_state, next_itsx_state)

            # print("update ",time.time()-update_start)
            if self.replace_count % self.replace_target_iter == 0:
                print("replace para")
                # start_time=time.time()
                self.replace_para(self.target_model.variables, self.eval_model.variables)
                # print("replace cost",time.time()-start_time)
        # update epsilon
        fraction = min(float(self.learn_count) / self.decay_steps, 1)
        self.epsilon = self.max_epsilon + fraction * (self.min_epsilon - self.max_epsilon)

        if self.epsilon < 0.001:
            self.epsilon = 0.001

    @tf.function
    def replace_para(self, target_var, source_var):
        for (a, b) in zip(target_var, source_var):
            a.assign(a * (1 - self.tau) + b * self.tau)

    @tf.function
    def update_gradient(self, ll_state, itsx_state, a, r, next_ll_state, next_itsx_state):
        with tf.GradientTape() as tape:
            q_eval = self.eval_model([ll_state, self.ll_adj_batch, itsx_state, self.itsx_adj_batch])
            # reshape to (batch,branch,subaction)
            q_target = q_eval.numpy().copy()
            # mask construction
            eval_act_index = a.numpy().astype(int)  # action index of experience
            eval_act_index_mask = tf.one_hot(eval_act_index, self.subaction_num)
            eval_act_index_reverse_mask = tf.one_hot(eval_act_index, self.subaction_num, on_value=0.0,
                                                     off_value=1.0)  # we need reversed one hot matrix to reserve the value of not eval action index

            q_target = tf.multiply(q_target, eval_act_index_reverse_mask)  # remove the value on eval index

            # on policy estimate next value
            q_next_eval = self.eval_model(
                [next_ll_state, self.ll_adj_batch, next_itsx_state, self.itsx_adj_batch]).numpy().copy()
            greedy_next_action = np.argmax(q_next_eval, axis=2)  # get greedy action index of next eval
            greedy_next_action_mask = tf.one_hot(greedy_next_action,
                                                 self.subaction_num)  # mask to select only correpsond q value

            q_target_s_next = self.target_model([next_ll_state, self.ll_adj_batch, next_itsx_state,
                                                 self.itsx_adj_batch]).numpy().copy()  # compute with target network estimate
            masked_q_target_s_next = tf.multiply(q_target_s_next,
                                                 greedy_next_action_mask)  # selected q value based on on-policy greedy action
            if self.td_operator_type == 'MEAN':
                operator = tf.reduce_sum(masked_q_target_s_next, 2)  # compute greedy
                operator = tf.multiply(operator[:, tf.newaxis], self.region_reward_mask)
                operator = tf.reduce_mean(operator, axis=2)
                operator = tf.multiply(tf.expand_dims(operator, -1), eval_act_index_mask)
            elif self.td_operator_type == "MAX":
                operator = tf.reduce_sum(masked_q_target_s_next, 2)
                operator = tf.expand_dims(tf.reduce_max(operator, 1), -1)
                operator = tf.multiply(tf.expand_dims(operator, -1), eval_act_index_mask)
                # operator=tf.reduce_max(operator)
            elif self.td_operator_type == "NAIVE":
                operator = tf.reduce_sum(masked_q_target_s_next, 2)
                operator = tf.multiply(tf.expand_dims(operator, -1), eval_act_index_mask)
            else:
                raise Exception("unknown operator type")

            masked_r = tf.multiply(tf.expand_dims(r, -1), eval_act_index_mask)
            q_target = q_target + masked_r + self.gamma * operator  # fill the q_target value of eval act
            # only replace the value of action index in experience
            loss = tf.keras.losses.mean_squared_error(q_eval, tf.convert_to_tensor(q_target))

            self.loss_his.append(np.average(loss.numpy()))

            # print(np.average(loss.numpy()))
        gradients = tape.gradient(loss, self.eval_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_model.trainable_variables))

    def store_transition(self, ll_state, itsx_state, a, r, next_ll_state, next_itsx_state):
        index = self.memory_counter % self.memory_size  # store transition at position ,first coming first replaced.
        self.ll_state_memory[index] = ll_state
        self.itsx_state_memory[index] = itsx_state
        self.action_memory[index] = a
        self.reward_memory[index] = r
        self.next_ll_state_memory[index] = next_ll_state
        self.next_itsx_state_memory[index] = next_itsx_state
        self.memory_counter += 1

    def update_support_info(self, ll_adj, itsx_adj, itsx_order, assignment):
        self.ll_adj = ll_adj[np.newaxis, :]
        self.itsx_adj = itsx_adj[np.newaxis, :]
        self.ll_adj_batch = tf.repeat(self.ll_adj, [self.batch_size], axis=0)
        self.itsx_adj_batch = tf.repeat(self.itsx_adj, [self.batch_size], axis=0)
        self.region_reward_mask = []
        for assign in assignment:
            activated_indices = np.hstack([np.where(np.array(itsx_order) == itsx_id) for itsx_id in np.array(assign)])
            current_mask = np.zeros(len(itsx_order))
            current_mask[activated_indices] = 1
            for _ in range(len(assign)):
                self.region_reward_mask.append(current_mask)
        self.region_reward_mask = np.array(self.region_reward_mask, dtype=np.int32)

    def save_model(self, episode):
        """
        model file type is .h5
        :param model_folder: relative path to model folder
        :return:
        """
        model_path = os.path.join(self.work_folder, "agent_model", "agent_" + str(self.agent_id),
                                  "episode_" + str(episode))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.eval_model.save(os.path.join(model_path, "eval_model.h5"))


class Attention(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.nn.relu, l2=0.0):
        super(Attention, self).__init__()

        self.l2 = l2
        self.activation = activation
        self.units = units

    def build(self, input_shape):
        H_shape, A_shape = input_shape

        self.W = self.add_weight(
            shape=(H_shape[2], self.units),
            initializer='glorot_uniform',
            dtype=tf.float32,
            regularizer=tf.keras.regularizers.l2(self.l2)
        )

        self.a_1 = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            dtype=tf.float32,
            regularizer=tf.keras.regularizers.l2(self.l2)
        )

        self.a_2 = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            dtype=tf.float32,
            regularizer=tf.keras.regularizers.l2(self.l2)
        )

    def call(self, inputs):
        H, A = inputs
        X = H @ self.W

        attn_self = X @ self.a_1
        attn_neighbours = X @ self.a_2

        attention = attn_self + tf.transpose(attn_neighbours, perm=[0, 2, 1])

        E = tf.nn.leaky_relu(attention)

        mask = -10e9 * (1.0 - A)
        masked_E = E + mask

        # A = tf.cast(tf.math.greater(A, 0.0), dtype=tf.float32)

        alpha = tf.nn.softmax(masked_E)

        H_cap = alpha @ X

        out = self.activation(H_cap)

        return out


class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, output_layer=False, activation=tf.identity, l2=0.0, name="g"):
        super(GraphAttentionLayer, self).__init__(name=name)

        self.activation = activation
        self.num_heads = num_heads
        self.output_layer = output_layer

        self.attn_layers = [Attention(units, l2=l2) for x in range(num_heads)]

    def call(self, inputs):

        H, A = inputs

        H_out = [self.attn_layers[i]([H, A]) for i in range(self.num_heads)]

        if self.output_layer:
            multi_head_attn = tf.reduce_mean(tf.stack(H_out), axis=0)
            multi_head_attn = tf.keras.layers.Flatten()(multi_head_attn)
            out = self.activation(multi_head_attn)
        else:
            multi_head_attn = tf.concat(H_out, axis=-1)
            out = self.activation(multi_head_attn)

        return out


def build_network(intersection_num, state_dim, sub_action_num, region_assignment, cell_length=5,head_num=8):
    print("build Branching DQ network")
    print("Lane level aggregation")
    lane_count = intersection_num * 12
    lane_level_input = layers.Input(shape=(lane_count, cell_length), name="lane_level_state")
    lane_level_adj_matrix = layers.Input(shape=(lane_count, lane_count), name="lane_level_adj_matrix")
    # ll_embedding=layers.Dense(32, activation="relu")(lane_level_input)
    ll_attention_encoding = GraphAttentionLayer(units=8, num_heads=head_num, activation=tf.nn.elu, l2=0.0005,
                                                name="lane_level_encoding_1")(
        [lane_level_input, lane_level_adj_matrix])
    ll_attention_encoding = GraphAttentionLayer(units=16, num_heads=head_num, activation=tf.nn.elu,
                                                l2=0.0005, name="lane_level_encoding_2")(
        [ll_attention_encoding, lane_level_adj_matrix])

    print("Intersection-level aggregation")
    itsx_level_input = layers.Input(shape=(intersection_num, state_dim), name="itsx_level_state")  # input layer
    adjacent_matrix = layers.Input(shape=(intersection_num, intersection_num), name="itsx_level_adj_matrix")
    # itsx_embedding=layers.Dense(32, activation="relu")(itsx_level_input)
    attention_embeding = GraphAttentionLayer(units=8, num_heads=head_num, activation=tf.nn.elu, l2=0.0005,
                                             name="itsx_level_encoding_1")(
        [itsx_level_input, adjacent_matrix])

    attention_embeding = GraphAttentionLayer(units=16, num_heads=head_num, activation=tf.nn.elu, l2=0.0005,
                                             name="itsx_level_encoding_2")(
        [attention_embeding, adjacent_matrix])

    print("Concatenating ll-level with itsx-level and build BDQ for each region")
    output_layer = []
    region_l_count_t = int(lane_count / 4)
    start_lane_id = 0
    start_itsx_id = 0
    for i, assignment in enumerate(region_assignment):
        region_l_count = len(assignment) * 12
        end_lane_id = start_lane_id + region_l_count
        end_itsx_id = start_itsx_id + len(assignment)
        # assert region_l_count_t==region_l_count
        # assert start_lane_id == (i * region_l_count) and start_itsx_id == i * 4
        # assert end_lane_id == (i + 1) * region_l_count and end_itsx_id == (i + 1) * 4
        current_region_ll_level = layers.Flatten()(ll_attention_encoding[:, start_lane_id:end_lane_id, :])
        current_region_itsx_level = layers.Flatten()(attention_embeding[:, start_itsx_id:end_itsx_id, :])
        start_lane_id = end_lane_id
        start_itsx_id = end_itsx_id

        concated_layer = layers.Concatenate()([current_region_ll_level, current_region_itsx_level])
        # build shared representation hidden layer
        shared_representation = layers.Dense(1024, activation="relu")(concated_layer)
        shared_representation = layers.Dense(512, activation="relu")(shared_representation)
        # build common state value layer
        common_state_value = layers.Dense(256, activation="relu")(shared_representation)
        common_state_value = layers.Dense(1)(common_state_value)
        # build action branch q value layer iteratively
        subaction_q_layers = []

        for _ in range(len(assignment)):
            action_branch_layer = layers.Dense(256, activation="relu")(shared_representation)
            action_branch_layer = layers.Dense(sub_action_num)(action_branch_layer)

            subaction_q_value = common_state_value + (action_branch_layer - tf.reduce_mean(action_branch_layer))

            subaction_q_layers.append(subaction_q_value)
            # print(subaction_q_value.shape)
        subaction_q_layers = tf.stack(subaction_q_layers, axis=1)
        # print(subaction_q_layers.shape)
        output_layer.append(subaction_q_layers)
    output_layer = layers.Concatenate(axis=1)(output_layer)
    model = tf.keras.Model([lane_level_input, lane_level_adj_matrix, itsx_level_input, adjacent_matrix],
                           output_layer)

    return model


