import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import os
import sys
from collections import defaultdict, deque
from src.forecast.traffic_forecast import TrafficForecaster

class MarlEnv(gym.Env):
    """Multi-Agent Reinforcement Learning environment for traffic signal control using SUMO with predictive states.

    This environment manages multiple traffic lights as agents, incorporating traffic forecasting for state augmentation.
    """

    def __init__(self, config_path='configs/grid.sumocfg', min_green=5, max_green=60, yellow_time=3, reward_weights={'queue': -0.1, 'wait': -0.01, 'flicker': -1.0}, forecast_steps=5):
        # Detailed comment: Initialize the MARL environment.
        # Parameters:
        # - config_path: Path to SUMO configuration file.
        # - min_green: Minimum green time for a phase.
        # - max_green: Maximum green time for a phase.
        # - yellow_time: Duration of yellow phase.
        # - reward_weights: Weights for reward components.
        # - forecast_steps: Number of steps to forecast.
        self._validate_config(min_green, max_green, yellow_time, forecast_steps)
        self.config_path = config_path
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.reward_weights = reward_weights
        self.forecast_steps = forecast_steps
        self._start_sumo()  # Start SUMO before accessing traci
        try:
            self.intersections = traci.trafficlight.getIDList()
        except traci.exceptions.TraCIException as e:
            raise RuntimeError(f"Failed to get intersections: {e}")
        self.num_agents = len(self.intersections)
        self.action_space = [spaces.Discrete(2) for _ in self.intersections]
        base_shape = 8  # 4 directions * 2 (queue, wait) for self
        neighbor_shape = 8 * (self.num_agents - 1)
        predict_shape = self.forecast_steps * base_shape  # Predictions for self only
        obs_shape = base_shape + neighbor_shape + predict_shape
        self.observation_space = [spaces.Box(low=0, high=np.inf, shape=(obs_shape,)) for _ in self.intersections]
        self.current_phase = {tl: 0 for tl in self.intersections}
        self.phase_time = {tl: 0 for tl in self.intersections}
        self.is_yellow = {tl: False for tl in self.intersections}
        self.yellow_start = {tl: 0 for tl in self.intersections}
        self.neighbors = self._detect_neighbors()
        self.edge_mapping = defaultdict(list)
        self.phase_defs = ['GGGGrrrrGGGGrrrr', 'yyyyrrrryyyyrrrr', 'rrrrGGGGrrrrGGGG', 'rrrryyyyrrrryyyy']
        self.forecaster = {tl: TrafficForecaster(input_timesteps=10, output_timesteps=forecast_steps, features=base_shape) for tl in self.intersections}
        self.history = {tl: deque(maxlen=10) for tl in self.intersections}  # For LSTM input
        self._init_edge_mapping()
        self._init_phases()

    def _validate_config(self, min_green, max_green, yellow_time, forecast_steps):
        # Detailed comment: Validate configuration parameters to ensure logical consistency.
        if min_green >= max_green:
            raise ValueError("min_green must be less than max_green")
        if yellow_time < 0 or forecast_steps < 1:
            raise ValueError("yellow_time and forecast_steps must be positive")

    def _detect_neighbors(self):
        # Detailed comment: Dynamically detect neighboring intersections for information sharing in MARL.
        neighbors = {}
        # Logic to detect neighbors based on connections (placeholder for actual implementation)
        return neighbors

    def _init_edge_mapping(self):
        # Detailed comment: Map each intersection to its incoming edges for state observation.
        for tl in self.intersections:
            controlled_links = traci.trafficlight.getControlledLinks(tl)
            edges = set()
            for link in controlled_links:
                if link:
                    edges.add(link[0][0].split('_')[0])
            self.edge_mapping[tl] = list(edges)

    def _init_phases(self):
        # Detailed comment: Set initial phases for all traffic lights.
        for tl in self.intersections:
            traci.trafficlight.setRedYellowGreenState(tl, self.phase_defs[0])

    def _start_sumo(self):
        # Detailed comment: Launch the SUMO simulation via TraCI.
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        sumo_cmd = [sumo_binary, '-c', self.config_path]
        traci.start(sumo_cmd)

    def _get_base_state(self, tl_id):
        # Detailed comment: Compute base state features (queue and wait times) for a given traffic light.
        state = []
        for edge in self.edge_mapping[tl_id]:
            queue = traci.edge.getLastStepHaltingNumber(edge)
            num_lanes = traci.edge.getLaneNumber(edge)
            wait = sum(traci.lane.getWaitingTime(f"{edge}_{i}") for i in range(num_lanes))
            state.extend([queue, wait])
        state = np.array(state)
        state = np.pad(state, (0, 8 - len(state) % 8), mode='constant')
        return state

    def _get_state(self, tl_id):
        # Detailed comment: Construct full observation including current state, neighbors, and forecasts.
        current = self._get_base_state(tl_id)
        self.history[tl_id].append(current)
        neighbor_state = []
        for neighbor in self.neighbors.get(tl_id, []):
            neighbor_state.extend(self._get_base_state(neighbor))
        try:
            prediction = self.forecaster[tl_id].predict(hist_array).flatten()
        except Exception as e:
            print(f"Prediction error for {tl_id}: {e}")
            prediction = np.zeros(self.forecast_steps * len(current))
        return np.concatenate([current, neighbor_state, prediction])

    def reset(self):
        # Detailed comment: Reset the simulation and clear histories. Returns initial observations for all agents.
        traci.close()
        self._start_sumo()
        self._init_phases()
        for hist in self.history.values():
            hist.clear()
        return [self._get_state(tl) for tl in self.intersections]

    def step(self, actions):
        # Detailed comment: Execute actions for all agents, update simulation, compute rewards. Returns next states, rewards, dones, infos.
        rewards = []
        dones = [False] * self.num_agents
        infos = [{}] * self.num_agents
        for i, tl in enumerate(self.intersections):
            action = actions[i]
            if self.is_yellow[tl]:
                if traci.simulation.getTime() - self.yellow_start[tl] >= self.yellow_time:
                    self.is_yellow[tl] = False
                    self.current_phase[tl] = (self.current_phase[tl] + 1) % len(self.phase_defs)
                    self.phase_time[tl] = 0
                    self._set_phase(tl, self.current_phase[tl])
            else:
                self.phase_time[tl] += 1
                if action == 1 and self.phase_time[tl] >= self.min_green:
                    self.is_yellow[tl] = True
                    self.yellow_start[tl] = traci.simulation.getTime()
                    self._set_phase(tl, (self.current_phase[tl] + 1) % len(self.phase_defs))
            rewards.append(self._compute_reward(tl))
        traci.simulationStep()
        next_states = [self._get_state(tl) for tl in self.intersections]
        return next_states, rewards, dones, infos

    def _set_phase(self, tl_id, phase):
        # Detailed comment: Apply the specified phase to the traffic light.
        traci.trafficlight.setRedYellowGreenState(tl_id, self.phase_defs[phase])

    def _compute_reward(self, tl_id):
        # Detailed comment: Calculate reward based on queue length, waiting time, and phase flickering.
        queue = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in self.edge_mapping[tl_id])
        wait = 0
        for edge in self.edge_mapping[tl_id]:
            num_lanes = traci.edge.getLaneNumber(edge)
            wait += sum(traci.lane.getWaitingTime(f"{edge}_{i}") for i in range(num_lanes))
        flicker = -1 if self.phase_time[tl_id] < self.min_green else 0
        return self.reward_weights['queue'] * queue + self.reward_weights['wait'] * wait + self.reward_weights['flicker'] * flicker

    def close(self):
        # Detailed comment: Terminate the SUMO connection.
        traci.close()