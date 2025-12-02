import numpy as np

class WebsterMethod:
    """Webster's Method for calculating optimal traffic signal cycle times."""

    def __init__(self, lost_time=5, saturation_flow=1800, min_cycle=60, max_cycle=180):
        """
        Initialize Webster's Method parameters.
        :param lost_time: Total lost time per cycle (seconds)
        :param saturation_flow: Saturation flow rate (vehicles/hour/lane)
        :param min_cycle: Minimum cycle length (seconds)
        :param max_cycle: Maximum cycle length (seconds)
        """
        self.lost_time = lost_time
        self.saturation_flow = saturation_flow
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle

    def calculate_cycle_length(self, flow_ratios):
        """
        Calculate optimal cycle length using Webster's formula.
        :param flow_ratios: List of critical flow ratios for each phase
        :return: Optimal cycle length (seconds)
        """
        y = sum(flow_ratios)
        if y >= 1:
            return self.max_cycle  # Over-saturated, use max cycle
        cycle_length = (1.5 * self.lost_time + 5) / (1 - y)
        return max(self.min_cycle, min(self.max_cycle, cycle_length))

    def calculate_green_times(self, cycle_length, flow_ratios):
        """
        Allocate green times proportionally to flow ratios.
        :param cycle_length: Total cycle length (seconds)
        :param flow_ratios: List of flow ratios for each phase
        :return: List of green times for each phase
        """
        effective_green = cycle_length - self.lost_time
        total_ratio = sum(flow_ratios)
        green_times = [int((ratio / total_ratio) * effective_green) for ratio in flow_ratios]
        return green_times

    def get_action(self, state):
        """
        Get signal control action based on current state.
        :param state: Current traffic state (e.g., vehicle counts per lane)
        :return: Dictionary with phase green times
        """
        # Assuming state is a dict with 'volumes' for each phase
        volumes = state.get('volumes', [0] * 4)  # Example for 4 phases
        flow_ratios = [v / self.saturation_flow for v in volumes]
        cycle_length = self.calculate_cycle_length(flow_ratios)
        green_times = self.calculate_green_times(cycle_length, flow_ratios)
        return {'cycle_length': cycle_length, 'green_times': green_times}

# Example usage
if __name__ == '__main__':
    webster = WebsterMethod()
    sample_ratios = [0.2, 0.3, 0.15, 0.25]
    cycle = webster.calculate_cycle_length(sample_ratios)
    greens = webster.calculate_green_times(cycle, sample_ratios)
    print(f'Optimal Cycle: {cycle}s, Green Times: {greens}')