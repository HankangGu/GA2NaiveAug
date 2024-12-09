import cityflow as cf
import networkx as nx
import json

import numpy as np
import os
import pickle


# This class is inspired from

class Intersection:
    """
    this class stores necessary connectivity property for one intersections

    """

    def __init__(self, config):
        """
        initialize intersection object
        :param config: raw info for this itsx
        """

        # self.config = config
        self.id = config['id']  # itsx id
        self.enter_roads = []  # road(from other itsx) that enter this itsx
        self.enter_lanes = []
        self.leave_roads = []  # road that leave this itsx
        self.leave_lanes = []
        self.neighbour_itsx = []
        self.x = int(self.id.split('_')[1])  # get x coordintate of this itsx
        self.y = int(self.id.split('_')[2])  # get y coordinate of this itsx
        for road_id in config['roads']:  # iterate through all itsx
            id_chunk = road_id.split('_')  # identify from which itsx it leaves
            road_x = int(id_chunk[1])
            road_y = int(id_chunk[2])
            if road_x == self.x and road_y == self.y:
                # road share same x and y as intersection so this road leaves this intersection
                self.leave_roads.append(road_id)
                for lane_id in range(3):
                    self.leave_lanes.append(road_id + "_" + str(lane_id))
            else:
                # otherwise this road enters itsx
                temp_itsx = "intersection_" + str(road_x) + "_" + str(road_y)
                if self.neighbour_itsx.__contains__(temp_itsx):
                    self.neighbour_itsx.append(temp_itsx)
                self.enter_roads.append(road_id)
                for lane_id in range(3):
                    self.enter_lanes.append(road_id + "_" + str(lane_id))
        self.movement = {}  # key: enter lane; value: leave lane
        self.movement_lane_lane={}
        self.leave_road_to_enter_lane = {leave_road_id: [] for leave_road_id in
                                         self.leave_roads}  # key: leave road id ; value: lanes that enter this road
        self.enter_road_to_its_enter_lane = {enter_road_id: [] for enter_road_id in self.enter_roads}

        for roadlink in config['roadLinks']:  # roadLinks defines the movement between roads
            # specify the enter lanes for each road
            if roadlink['type'] == 'go_straight':
                lane_suffix = 1
            elif roadlink['type'] == 'turn_left':
                lane_suffix = 0
            elif roadlink['type'] == 'turn_right':
                lane_suffix = 2
            else:
                raise ("unknown movement")

            self.movement[roadlink['startRoad'] + "_" + str(lane_suffix)] = roadlink['endRoad'] + "_" + str(lane_suffix)
            # self.movement_lane_lane
            self.leave_road_to_enter_lane[roadlink['endRoad']].append(roadlink['startRoad'] + "_" + str(lane_suffix))
        # build green phase to incoming lane
        self.phase_to_lane = {}
        for phase_id, phase_config in enumerate(config['trafficLight']['lightphases']):
            self.phase_to_lane[phase_id] = {}
            green_movement = phase_config['availableRoadLinks']
            self.phase_to_lane[phase_id]['green'] = []
            for link_id in green_movement:
                roadlink = config['roadLinks'][link_id]
                if roadlink['type'] == 'go_straight':
                    lane_suffix = 1
                elif roadlink['type'] == 'turn_left':
                    lane_suffix = 0
                elif roadlink['type'] == 'turn_right':
                    lane_suffix = 2
                self.phase_to_lane[phase_id]['green'].append(roadlink['startRoad'] + "_" + str(lane_suffix))

        """------------dynamic property------------------"""
        self.waiting_queue_per_lane = {key: 0 for key in self.enter_lanes}
        self.pressure = 0
        self.wave_per_lane = {key: 0 for key in self.enter_lanes}


    def _update_wave(self, lane_vehicle_count):
        for in_lane_id in self.enter_lanes:  # iterate through all incoming roads
            self.wave_per_lane[in_lane_id] = lane_vehicle_count[in_lane_id]

    def get_wave(self):
        return self.wave_per_lane

    def get_waiting_queue(self):
        return self.waiting_queue_per_lane

    def get_pressure(self):
        return self.pressure

    def neighbour_relation_build(self, neighbour_info_list):
        """

        :param neighbour_info_list:
        :return:
        """

        def _update(neighbour_info):
            for enter_road, corresponding_lane_list in neighbour_info.items():
                if enter_road in self.enter_roads_nei_lane.keys():
                    self.enter_roads_nei_lane[enter_road] = corresponding_lane_list

        self.enter_roads_nei_lane = {key: ['dummy' for _ in range(3)] for key in self.enter_roads}
        for neighbour_info in neighbour_info_list:
            if not neighbour_info == "dummy":
                _update(neighbour_info)


class CityFlowEnvWrapper:
    """
    cityflow environment demo

    phase cycle

    1->3->2->4 with all stop phase 0 between consecutive phase


    """

    # TODO decide whether leave getstate to Intersection obj or Env
    # TODO decide whether return a list of state or a dict
    def __init__(self, ENV_CONFIG):
        """
        intialise member variables and write configuration

        :param config_path:
        :param partition_config: hops, stride, overlap, full_cover
        """

        self.env_config_dict = ENV_CONFIG
        # default

        # roadnet_path = ENV_CONFIG["ROADNET_PATH"]

        self.intersections = dict()
        self.lane_length = dict()
        self.roadnet = json.load(
            open(os.path.join(self.env_config_dict["PATH_TO_WORK_DIRECTORY"], ENV_CONFIG["ROADNET_PATH"])))
        #
        self.intersection_ids = []  # intersection with four roads
        for itsx in self.roadnet['intersections']:
            if len(itsx['roads']) == 8:
                self.intersection_ids.append(itsx['id'])
                self.intersections[itsx['id']] = Intersection(itsx)
        for road in self.roadnet['roads']:
            start_point = road['points'][0]
            end_point = road['points'][1]
            road_length = abs(start_point['x'] - end_point['x']) + abs(start_point['y'] - end_point['y'])
            for lane_suffix in range(len(road['lanes'])):
                lane_id = road['id'] + "_" + str(lane_suffix)
                self.lane_length[lane_id] = road_length

            # self.lane_length[road_id]=
        # config neighbour info for each itsx
        for itsx_id, itsx in self.intersections.items():
            x = itsx.x
            y = itsx.y

            neighbour_co = [[0, 1], [-1, 0], [1, 0], [0, -1]]
            neighbour_info = []
            for x_incre, y_incre in neighbour_co:
                neighbour_id = "intersection_" + str(x + x_incre) + "_" + str(y + y_incre)
                if neighbour_id in self.intersection_ids:
                    neighbour_info.append(self.intersections[neighbour_id].leave_road_to_enter_lane)
                else:
                    neighbour_info.append("dummy")
            itsx.neighbour_relation_build(neighbour_info)
        self.flow_file = json.load(
            open(os.path.join(self.env_config_dict['PATH_TO_WORK_DIRECTORY'], self.env_config_dict['FLOW_PATH']), 'r'))
        self.action_type = ENV_CONFIG["ACTION_TYPE"]
        self.yellow_duration = ENV_CONFIG["YELLOW_DURATION"]
        self.expected_enter_vehicle_traj = self._build_expected_enter_vehicle_traj()
        # self.vehicle_enter_leave_dict = dict()
        # self.previous_vehicles_list = {}
        # self.intersections_phase_his= []
        self.state_return_type = ENV_CONFIG["STATE_RETURN_TYPE"]
        self.cell_count=ENV_CONFIG["CELL_COUNT"]
        self.log = False  # only log after convergence

    def reset(self, round):
        """
        reset env
        :param round:
        :return:
        """
        # print(cityflow_config)
        self.round = round
        # log last round control info
        if self.round > 1500:
            # log after converge
            for itsx, track in self.phase_track.items():
                log_file_name = str(itsx) + '.txt'
                log_file_path = os.path.join(self.log_folder, log_file_name)
                with open(log_file_path, 'w') as f:
                    f.write('\n'.join(track))

        # prepare for next round
        self.phase_track = {key: [] for key in self.intersection_ids}
        self.log_folder = os.path.join(self.env_config_dict["PATH_TO_WORK_DIRECTORY"], "roundlog", str(round))
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        if self.env_config_dict["LOG"] and round > 1500:
            a = os.path.join(self.env_config_dict["PATH_TO_WORK_DIRECTORY"], "roundlog", str(round - 1))
            with open(os.path.join(a, "phase_log.pickle"), 'wb') as f:
                pickle.dump(self.intersections_phase_his, f)

            self.env_config_dict["SAVEREPLAY"] = True

        cityflow_config = {
            "interval": self.env_config_dict["INTERVAL"],
            "seed": 0,
            "laneChange": False,
            "dir": self.env_config_dict["PATH_TO_WORK_DIRECTORY"] + "/",
            "roadnetFile": self.env_config_dict["ROADNET_PATH"],
            "flowFile": self.env_config_dict["FLOW_PATH"],
            "rlTrafficLight": self.env_config_dict["RLTRAFFICLIGHT"],
            "saveReplay": self.env_config_dict["SAVEREPLAY"],
            "roadnetLogFile": os.path.join("roundlog", str(round), "roadnetLogFile.json"),
            "replayLogFile": os.path.join("roundlog", str(round), "replayLogFile.txt"),
        }
        # write file
        # self.config_path = os.path.join(self.env_config_dict["PATH_TO_WORK_DIRECTORY"], "cityflow.config")
        self.config_path = os.path.join(self.log_folder, "cityflow.config")
        with open(self.config_path, "w") as json_file:
            json.dump(cityflow_config, json_file)

        if round == 0:
            print(cityflow_config)
            print("=========================")
        # write file
        self.last_waiting_count = [0 for _ in range(len(self.intersection_ids))]
        self.current_phases = {key: 0 for key in self.intersection_ids}
        self.eng = cf.Engine(self.config_path)
        self.current_phases = {key: 0 for key in self.intersection_ids}
        self.vehicle_enter_leave_dict = dict()
        self.previous_vehicles_list = {}
        self.intersections_phase_his = []
        self.expected_moving_vehicle = set()
        self.current_phases_duration = {key: 0 for key in self.intersection_ids}

        # return {inter_id: np.zeros(25) for inter_id in self.intersection_ids}
        return self._get_state()

    def step(self, actions):
        """
          perform action
        -> collect state and reward
        :param actions:
        :return:
        TODO return state of different type
        """
        current_time = self.eng.get_current_time()
        current_timestep = int(
            current_time / (self.env_config_dict['INTERVAL'] * self.env_config_dict['ACTION_INTERVAL']))
        additional_log = {"pressure": {key: 0 for key in self.intersection_ids},
                          "queuelen": {key: 0 for key in self.intersection_ids},

                          }

        # assign actions to each intersections
        # for itsx, action in actions.items():
        #     if self.action_type == "SWITCH" or self.action_type == "twoPhaseAllPass":
        #         if action == 1:
        #             current_phase = self.current_phases[itsx]
        #             self.eng.set_tl_phase(itsx, self._next_phase(current_phase))
        #             self.current_phases[itsx] = self._next_phase(current_phase)
        #     elif self.action_type == "CHOOSE PHASE":
        #         self.eng.set_tl_phase(itsx, action)
        #         self.current_phases[itsx] = action

        # update the signals changed to a new phase
        changed_itsx_signal = {}  # key-> itsx; value -> new phase
        for itsx, action in actions.items():
            self.phase_track[itsx].append(str(current_time) + " " + str(action))
            if self.action_type == "SWITCH" and action == 1:
                changed_itsx_signal[itsx] = self._next_phase(self.current_phases[itsx])
                self.current_phases[itsx] = self._next_phase(self.current_phases[itsx])
            elif self.action_type == "CHOOSE PHASE" and action != self.current_phases[itsx]:
                changed_itsx_signal[itsx] = action
                self.current_phases[itsx] = action
                self.current_phases_duration[itsx] = self.env_config_dict["ACTION_INTERVAL"]
            else:
                # no signal change for this itsx
                self.current_phases_duration[itsx] += self.env_config_dict["ACTION_INTERVAL"]
        self.intersections_phase_his.append(changed_itsx_signal)
        # self._update_expected_enter_vehicle()
        self.expected_moving_vehicle = self.expected_moving_vehicle.union(
            self.expected_enter_vehicle_traj[current_timestep])
        # simulate env for fixed step to receive actions for next
        for sec in range(self.env_config_dict["ACTION_INTERVAL"]):
            # set signal phase for each simulation step
            # for sec< Yellow duration, red phase should be inserted between two different phase
            if len(changed_itsx_signal) > 0:
                if self.yellow_duration > 0 and sec == 0:  # red phase
                    plan = {key: 0 for key in changed_itsx_signal.keys()}
                    self._set_signals(plan)
                elif sec == self.yellow_duration:
                    self._set_signals(changed_itsx_signal)

            self.eng.next_step()
            self._update_enter_leave_time()
        # if len(self.expected_moving_vehicle - set(self.eng.get_vehicles())) > 0:
        #     print()
        self.expected_moving_vehicle = self.expected_moving_vehicle - set(self.eng.get_vehicles())
        # if len(self.expected_moving_vehicle) > 0:
        #     group these blocked vehicles into intersections according to the end of the first road in their routes

        # print("vehicles blocked")
        # collect reward of last state action and state for next step

        state = self._get_state()
        reward = self._get_reward()

        # intersection_congestion=self._get_itsx_congestion()
        #
        # additional_log["itsx_congestion"]=intersection_congestion
        if self.eng.get_vehicle_count() == 0:
            done = True
        else:
            done = False
        return state, reward, done, additional_log

    def _build_expected_enter_vehicle_traj(self):
        """
        build expected enter vehicle queue in each simulation period
        :return:
        """
        traj = [set() for _ in
                range(int(self.env_config_dict['SIM_TIMESPAN'] / self.env_config_dict['ACTION_INTERVAL']) + 1)]

        for flow_id, v in enumerate(self.flow_file):
            traj[
                int(v["startTime"] / (self.env_config_dict['INTERVAL'] * self.env_config_dict['ACTION_INTERVAL']))].add(
                "flow_" + str(flow_id) + "_0")
        # print(v)
        # for
        # start_timestep = step * self.env_config_dict['INTERVAL'] * self.env_config_dict['ACTION_INTERVAL']
        # end_timestep = (step + 1) * self.env_config_dict['INTERVAL'] * self.env_config_dict['ACTION_INTERVAL']

        # print()
        return traj

    # def _get_out_boundary_vehicles(self):
    #     return self.expected_moving_vehicle-set(self.eng.get_vehicles())

    def _get_itsx_congestion(self):
        def _sum_road_congestion(road_id, lane_count):
            sum = 0
            for i in range(3):
                sum += lane_count[road_id + "_" + str(i)]
            return sum

        itsx_congestion = {}
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        for itsx_id, itsx in self.intersections.items():
            c = 0
            for enter_road in itsx.enter_roads:
                c += _sum_road_congestion(enter_road, lane_vehicle_count)
            itsx_congestion[itsx_id] = c
        return itsx_congestion

    def _set_signals(self, itsx_phase_dict):
        for itsx, phase in itsx_phase_dict.items():
            self.eng.set_tl_phase(itsx, phase)

    def _get_state(self):
        """
        collect measurements according to the configuration file
        :return: {} key->itsx; value->{}: key->measurement name, value-> measurement value
        """
        state = {}
        state_temp = {}
        # lane_vehicles = self.eng.get_lane_vehicles()
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        blocked_vehicle_count = self._group_blocked_vehicles_count()
        # travel_distance = self.eng.get_vehicle_distance()
        for id in self.intersection_ids:
            if self.state_return_type == "RAW":
                # state[id] = {"waiting_queue_length": self._collect_waiting_queue(id, lane_waiting_vehicle_count),
                #              "wave": self._collect_wave(id, lane_vehicle_count),
                #              "current_phase": [self.current_phases[id]],
                #              }
                state[id] = {"waiting_queue_length": self._collect_waiting_queue(id, lane_waiting_vehicle_count),
                             "wave": self._collect_wave(id, lane_vehicle_count, blocked_vehicle_count),

                             # "current_phase": [self.current_phases[id]],
                             "discretized_wave": self._collect_discretized_count(id)
                             # "green_waiting": [self._collect_green_lane_waiting_count(id,lane_waiting_vehicle_count)],
                             # "red_waiting": [sum(self._collect_waiting_queue(id, lane_waiting_vehicle_count))-self._collect_green_lane_waiting_count(id,lane_waiting_vehicle_count)],
                             # "current_phase_duration":[self.current_phases_duration[id]]
                             # "waiting_queue_length_neibor_lane": self._collect_waiting_queue_nei(id,
                             #                                                                     lane_waiting_vehicle_count),
                             # "neighbour_current_action": self._get_neighbour_policy(id)
                             # "cell_vector": self._collect_discretized_count(id)

                             }
                # state_temp[id] =
            else:
                # temp_state=self.augment_waiting_queue(id)
                temp_state = self._collect_waiting_queue(id, lane_waiting_vehicle_count)
                temp_wave = self._collect_wave(id, lane_vehicle_count)
                temp_state.extend(temp_wave)
                # temp_state=self.state_collector.collect_wave_outlane(id)
                # temp_wave_in_segement=self.state_collector.collect_wave_enter_segement(id,lane_vehicles,travel_distance)
                # if np.sum(temp_wave)!= np.sum(temp_wave_in_segement):
                #     print("invalid")
                # temp_state.extend(temp_wave_in_segement)
                temp_state.append(self.current_phases[id])
                state[id] = np.array(temp_state)
                # new_state = {}
                # for itsx_id, raw_state in state.items():
                #     temp_state = []
                #     for state_type, state_value in raw_state.items():
                #         temp_state.extend(state_value)
                #     new_state[itsx_id] = np.array(temp_state)
                # state = new_state

        return state

    def _get_neighbour_policy(self, itsx_id):
        neighbour_policy = []
        neigh_itsxs = self.intersections[itsx_id].neighbour_itsx
        for itsx in neigh_itsxs:
            neighbour_policy.append(self.current_phases[itsx])
        return neighbour_policy

    def _collect_waiting_queue_nei(self, intersection_id, lane_waiting_vehicle_count):
        """
        collect the number of waiting vehicles that will move on the incoming lanes of itsx
        :param intersection_id:
        :param lane_waiting_vehicle_count:
        :return:
        """
        incoming_lanes = self.intersections[intersection_id].enter_roads_nei_lane
        # waiting_queue_nei={key:[] for key in incoming_lanes.keys()}
        waiting_queue_nei = []
        # for road_id, nei_lane in incoming_lanes.items():
        #     for lane in nei_lane:
        #         if lane =='dummy':
        #             waiting_queue_nei[road_id].append(0)
        #         else:
        #             waiting_queue_nei[road_id].append(lane_waiting_vehicle_count[lane])
        for road_id, nei_lane in incoming_lanes.items():
            for lane in nei_lane:
                if lane == 'dummy':
                    waiting_queue_nei.append(0)
                else:
                    waiting_queue_nei.append(lane_waiting_vehicle_count[lane])
            # print(road_id)

        return waiting_queue_nei

    def _collect_waiting_queue(self, intersection_id, lane_waiting_vehicle_count):
        """
        this function collects waiting queue length on each lane of each intersection
        :param eng: cityflow environment
        :param intersection_id: the target intersection
        :return: a dictionary key: intersection_id; value: intersection state
        """
        in_roads_id = self.intersections[intersection_id].enter_roads  # get roads that enter target intersection

        waiting_queue = []
        for in_road_id in in_roads_id:  # iterate through all incoming roads

            for lane_index in range(3):  # each road has three lanes
                lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                waiting_queue.append(lane_waiting_vehicle_count[lane_id])  # collect waiting queue on each lane

        return waiting_queue

    def _collect_green_lane_waiting_count(self, intersection_id, lane_waiting_vehicle_count):
        in_lane_id = self.intersections[intersection_id].phase_to_lane[self.current_phases[intersection_id]]['green']
        c = 0
        for lane in in_lane_id:
            c += lane_waiting_vehicle_count[lane]
        return c

    def _collect_wave(self, intersection_id, lane_vehicle_count, blocked_vechile_count):
        """
        including waiting vehicles and moving vehicles
        :param intersection_id
        :return: a list of number of vehicle on incoming roads.
        """
        in_roads_id = self.intersections[intersection_id].enter_roads  # get roads that enter target intersection

        wave_count = []
        for in_road_id in in_roads_id:  # iterate through all incoming roads
            blocked_count = 0  # self._get_blocked_vehicles_count(intersection_id, blocked_vechile_count)
            for lane_index in range(3):  # each road has three lanes
                lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                wave_count.append(lane_vehicle_count[lane_id] + blocked_count / 3)  # collect waiting queue on each lane

        return wave_count

    def _collect_discretized_count(self, intersection_id):
        cell_num = self.cell_count  # how many cells during discretization
        lane_vehicles_id = self.eng.get_lane_vehicles()  # lane_id -> vehicle_id on the lane
        lane_vehicles_distance = self.eng.get_vehicle_distance()  # vehicle_id -> distance moved on lane
        in_roads_id = self.intersections[intersection_id].enter_roads
        cell_list = {}
        for in_road_id in in_roads_id:  # iterate through all incoming roads
            for lane_index in range(3):  # each road has three lanes
                lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                c = np.zeros(cell_num)
                cell_length = self.lane_length[lane_id] / cell_num
                current_lane_v_id = lane_vehicles_id[lane_id]
                for v in current_lane_v_id:
                    c[int(lane_vehicles_distance[v] / cell_length)] += 1
                cell_list[lane_id]=c
            # cell_list[in_road_id] = np.array(road_cell)
        return cell_list

    def _get_blocked_vehicles_count(self, intersection_id, blocked_vehicle_count):
        """

        :return:
        """
        in_roads_id = self.intersections[intersection_id].enter_roads
        count = 0
        for road_id in in_roads_id:
            if blocked_vehicle_count.__contains__(road_id):
                count += len(blocked_vehicle_count[road_id])
        return count

    def _group_blocked_vehicles_count(self):
        """


        :return:
        """
        ans = {}
        for v in list(self.expected_moving_vehicle):
            flow_id = int(v.split("_")[1])
            start_road = self.flow_file[flow_id]['route'][0]
            if ans.__contains__(start_road):
                ans[start_road].append(flow_id)
            else:
                ans[start_road] = [flow_id]
        return ans

    def _get_wave_count_quar(self, intersection_id, lane_vehicle_count):

        in_roads_id = self.intersections[intersection_id].enter_roads  # get roads that enter target intersection

        wave_count = 0
        for in_road_id in in_roads_id:  # iterate through all incoming roads
            for lane_index in range(3):  # each road has three lanes
                lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                if lane_vehicle_count[lane_id] > 7:
                    wave_count += lane_vehicle_count[lane_id]  # collect waiting queue on each lane

        return wave_count

    def _get_reward(self):
        reward = {}
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()

        for id in self.intersection_ids:
            reward[id] = self._get_queue_length(id, lane_waiting_vehicle_count)
        return reward

    # def _get_source_waiting_count(self):
    def _get_queue_length2(self, id, lane_waiting_vehicle_count):
        if not id is list:
            intersection_ids = [id]
        current_reward = []

        for intersection_id in intersection_ids:
            in_roads_id = self.intersections[intersection_id].enter_roads
            # leave_roads_id = self.intersections[intersection_id].leave_roads
            # in_roads_id.extend(leave_roads_id)
            for in_road_id in in_roads_id:
                for lane_index in range(3):  # each road has three lanes
                    lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                    # if lane_index==2 and lane_waiting_vehicle_count[lane_id]>0:
                    #     print(lane_id,"  ",lane_waiting_vehicle_count[lane_id])
                    current_reward.append(-lane_waiting_vehicle_count[lane_id])
        # current_reward = -current_reward
        return current_reward

    def _get_queue_length(self, id, lane_waiting_vehicle_count):
        if not id is list:
            intersection_ids = [id]
        current_reward = 0

        for intersection_id in intersection_ids:
            in_roads_id = self.intersections[intersection_id].enter_roads
            for in_road_id in in_roads_id:
                for lane_index in range(3):  # each road has three lanes
                    lane_id = in_road_id + '_' + str(lane_index)  # construct lane id
                    # if lane_index==2 and lane_waiting_vehicle_count[lane_id]>0:
                    #     print(lane_id,"  ",lane_waiting_vehicle_count[lane_id])
                    current_reward += lane_waiting_vehicle_count[lane_id]
        current_reward = -current_reward
        return current_reward

    def _get_max_pressure(self, intersection_ids, lane_vehicle_count):
        """
        w(l,m)=x(l)/x_{max}(l)-x(m)/x_{max}(m)
        :param intersection_ids: a list
        :return: total waiting vehicle count in
        """
        if not intersection_ids is list:
            intersection_ids = [intersection_ids]
        current_reward = 0
        for intersection_id in intersection_ids:
            intersection_pressure = 0
            for enter_lane_id, leave_lane_id in self.intersections[intersection_id].movement.items():
                # print(lane_id+"->"+out_lane_id)
                intersection_pressure += lane_vehicle_count[enter_lane_id] - lane_vehicle_count[
                    leave_lane_id]  # compute presure for this movement
            current_reward += abs(intersection_pressure)
        current_reward = -current_reward
        return current_reward

    def _update_enter_leave_time(self):
        """
        update enter leave time of each vehicle
        :return:
        """
        current_vehicles_list = self.eng.get_vehicles(include_waiting=True)
        enter_vehicles = set(current_vehicles_list) - set(self.previous_vehicles_list)
        current_time_step = self.eng.get_current_time()
        if len(enter_vehicles) > 0:
            # new enter
            # print()

            for v in enter_vehicles:
                # v has just enter the network
                self.vehicle_enter_leave_dict[v] = {"enter_time": current_time_step, "leave_time": None}
        leave_vehicles = set(self.previous_vehicles_list) - set(current_vehicles_list)
        if len(leave_vehicles) > 0:
            # some leave
            for v in leave_vehicles:
                self.vehicle_enter_leave_dict[v]["leave_time"] = current_time_step
        self.previous_vehicles_list = current_vehicles_list
        # print()

    def _next_phase(self, current_phase):
        current_phase = int(current_phase)
        if self.action_type == "twoPhaseAllPass":
            if current_phase == 9:
                return 10
            elif current_phase == 10:
                return 9
            else:
                raise Exception('wrong phase id')
        else:
            if current_phase == 1:
                return 3
            elif current_phase == 3:
                return 2
            elif current_phase == 2:
                return 4
            elif current_phase == 4:
                return 1
            else:
                raise Exception('wrong phase id')

    def save_replay(self, path):
        self.eng.set_replay_file(os.path.join(path, "replay.txt"))

    def get_average_travel_time(self):
        """

        :return: average travel time of vehicles that have completed their trip
        """

        return self.eng.get_average_travel_time()

    # def get_average_travel_time(self, a=1):
    #     total_travel_time = 0
    #     current_timestep = self.eng.get_current_time()
    #     for info in self.vehicle_enter_leave_dict.values():
    #         if info['leave_time'] is None:
    #             # vehicle has not left network in certain time step
    #             total_travel_time += current_timestep - info['enter_time']
    #         else:
    #             # vehicle has completed its trip
    #             total_travel_time += info['leave_time'] - info['enter_time']
    #     return total_travel_time / len(self.vehicle_enter_leave_dict.keys())

    def get_average_travel_time_verify(self):
        total_travel_time = 0
        finish_trip_count = 0
        # current_timestep = self.eng.get_current_time()
        for info in self.vehicle_enter_leave_dict.values():
            if not info['leave_time'] is None:
                # vehicle has completed its trip
                total_travel_time += info['leave_time'] - info['enter_time']
                finish_trip_count += 1
        return total_travel_time / finish_trip_count, finish_trip_count

    def get_throughput(self):
        finish_trip_count = 0
        for info in self.vehicle_enter_leave_dict.values():
            if not info['leave_time'] is None:
                # vehicle has completed its trip
                finish_trip_count += 1
        return finish_trip_count

    def get_intersections(self):
        return self.intersection_ids

    def get_max_distance(self):
        travel_distance = self.eng.get_vehicle_distance()
        distance = [value for _, value in travel_distance.items()]
        return np.max(distance)

    def get_average_queue_length(self):
        a = self.eng.get_lane_waiting_vehicle_count()
        sum = 0
        counter = 0
        for _, value in a.items():
            if value > 0:
                sum += value
                counter += 1
        return sum / (3 * 4 * 16)
    def get_movement_graph(self):
        G=nx.Graph()
        for itsx in self.intersections.values():
            for in_lane_id,out_lane_id in itsx.movement.items():
                G.add_edge(in_lane_id,out_lane_id)

        print(nx.adj_matrix)
        return G
    def get_movements(self):
        movements={}
        for id,itsx in self.intersections.items():
            movements[id]=itsx.movement
        return movements