import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import os
import sys
import time
import shutil
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
        """Validate configuration parameters to ensure logical consistency.
        
        Args:
            min_green: Minimum green phase duration
            max_green: Maximum green phase duration
            yellow_time: Yellow phase duration
            forecast_steps: Number of forecasting steps
            
        Raises:
            ValueError: If parameters are invalid or inconsistent
        """
        if min_green >= max_green:
            raise ValueError("min_green must be less than max_green")
        if yellow_time < 0 or forecast_steps < 1:
            raise ValueError("yellow_time and forecast_steps must be positive")

    def _detect_neighbors(self):
        """Dynamically detect neighboring intersections for information sharing in MARL.
        
        Returns:
            Dictionary mapping each intersection to its neighboring intersections
            
        Note:
            This is a placeholder implementation. In a real scenario, this would
            analyze the road network topology to identify physically connected
            intersections based on edge connectivity.
        """
        neighbors = {}
        # Logic to detect neighbors based on connections (placeholder for actual implementation)
        return neighbors

    def _init_edge_mapping(self):
        """Map each intersection to its incoming edges for state observation.
        
        Populates self.edge_mapping with traffic light IDs as keys and
        lists of controlled edge IDs as values. This mapping is used to
        collect traffic state information (queue lengths, waiting times)
        for each intersection.
        """
        for tl in self.intersections:
            controlled_links = traci.trafficlight.getControlledLinks(tl)
            edges = set()
            for link in controlled_links:
                if link:
                    edges.add(link[0][0].split('_')[0])
            self.edge_mapping[tl] = list(edges)

    def _init_phases(self):
        """Set initial phases for all traffic lights.
        
        Initializes all traffic lights to phase 0 (first green phase)
        to ensure consistent starting conditions across episodes.
        """
        for tl in self.intersections:
            traci.trafficlight.setRedYellowGreenState(tl, self.phase_defs[0])

    def _start_sumo(self):
        """Launch the SUMO simulation via TraCI.
        
        Handles SUMO binary detection, command construction, and connection
        establishment with robust error handling for existing connections.
        
        Raises:
            RuntimeError: If SUMO binary cannot be found or connection fails
        """
        # Resolve SUMO binary
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home:
            sumo_binary = os.path.join(sumo_home, 'bin', 'sumo')
        else:
            sumo_binary = shutil.which('sumo')

        if not sumo_binary:
            raise RuntimeError('SUMO binary not found (set SUMO_HOME or put sumo on PATH)')

        sumo_cmd = [sumo_binary, '-c', self.config_path]

        # Try to start SUMO, handling an existing connection if present.
        attempts = 3
        for attempt in range(attempts):
            try:
                traci.start(sumo_cmd)
                return
            except traci.exceptions.TraCIException as e:
                msg = str(e)
                # If a previous connection is active, try to close and retry
                if 'already active' in msg or 'already exists' in msg:
                    try:
                        traci.close()
                    except Exception:
                        pass
                    # Try clearing private connection registry if close() didn't work
                    if hasattr(traci, '_connections'):
                        try:
                            traci._connections.clear()
                        except Exception:
                            pass
                    if hasattr(traci, 'connection'):
                        try:
                            delattr(traci, 'connection')
                        except Exception:
                            pass
                    time.sleep(0.1)
                    continue
                # Other transient failures: wait and retry a couple times
                time.sleep(0.5)
        # Final attempt (let exception bubble if it fails)
        traci.start(sumo_cmd)

    def _get_base_state(self, tl_id):
        """Compute base state features (queue and wait times) for a given traffic light.
        
        Args:
            tl_id: Traffic light identifier
            
        Returns:
            numpy.ndarray: Base state vector with queue lengths and waiting times
                          for each direction, padded to ensure consistent size
        """
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
        """Construct full observation including current state, neighbors, and forecasts.
        
        Args:
            tl_id: Traffic light identifier
            
        Returns:
            numpy.ndarray: Complete observation vector containing:
                - Current traffic state (queue lengths, waiting times)
                - Neighboring intersections' states
                - Traffic forecasts for future steps
        """
        current = self._get_base_state(tl_id)
        self.history[tl_id].append(current)
        neighbor_state = []
        for neighbor in self.neighbors.get(tl_id, []):
            neighbor_state.extend(self._get_base_state(neighbor))
        try:
            if len(self.history[tl_id]) >= 10:
                hist_array = np.array(list(self.history[tl_id]))
                prediction = self.forecaster[tl_id].predict(hist_array).flatten()
            else:
                prediction = np.zeros(self.forecast_steps * len(current))
        except Exception:
            print(f"Prediction error for {tl_id}.")
            prediction = np.zeros(self.forecast_steps * len(current))
        return np.concatenate([current, neighbor_state, prediction])

    def reset(self):
        """Reset the simulation and clear histories.
        
        Returns:
            List[numpy.ndarray]: Initial observations for all agents
        """
        traci.close()
        self._start_sumo()
        self._init_phases()
        for hist in self.history.values():
            hist.clear()
        return [self._get_state(tl) for tl in self.intersections]

    def step(self, actions):
        """Execute actions for all agents, update simulation, compute rewards.
        
        Args:
            actions: List of actions for each agent (0=maintain, 1=switch phase)
            
        Returns:
            Tuple containing:
            - next_states: List of next observations for all agents
            - rewards: List of rewards for all agents
            - dones: List of episode termination flags
            - infos: List of additional information dictionaries
        """
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
        """Apply the specified phase to the traffic light.
        
        Args:
            tl_id: Traffic light identifier
            phase: Phase index to set (corresponds to phase_defs list)
        """
        traci.trafficlight.setRedYellowGreenState(tl_id, self.phase_defs[phase])

    def _compute_reward(self, tl_id):
        """Calculate reward based on queue length, waiting time, and phase flickering.
        
        Args:
            tl_id: Traffic light identifier
            
        Returns:
            float: Computed reward value (higher is better)
            
        Note:
            Reward components:
            - Queue penalty: Negative reward proportional to total queue length
            - Wait penalty: Negative reward proportional to total waiting time
            - Flicker penalty: Penalty for switching phases too frequently
        """
        queue = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in self.edge_mapping[tl_id])
        wait = 0
        for edge in self.edge_mapping[tl_id]:
            num_lanes = traci.edge.getLaneNumber(edge)
            wait += sum(traci.lane.getWaitingTime(f"{edge}_{i}") for i in range(num_lanes))
        flicker = -1 if self.phase_time[tl_id] < self.min_green else 0
        return self.reward_weights['queue'] * queue + self.reward_weights['wait'] * wait + self.reward_weights['flicker'] * flicker

    def close(self):
        """Terminate the SUMO connection.
        
        Cleanly closes the TraCI connection to SUMO simulation.
        Should be called when the environment is no longer needed.
        """
        traci.close()