import numpy as np

class FuzzySet:
    """Represents a fuzzy set with triangular membership function."""
    def __init__(self, low, mid, high):
        self.low = low
        self.mid = mid
        self.high = high

    def membership(self, x):
        """Compute membership degree for value x."""
        if x <= self.low or x >= self.high:
            return 0.0
        if x < self.mid:
            return (x - self.low) / (self.mid - self.low)
        return (self.high - x) / (self.high - self.mid)

class FuzzyRule:
    """Defines a fuzzy rule with antecedents and consequent."""
    def __init__(self, queue_level, output_timing):
        self.queue_level = queue_level
        self.output_timing = output_timing

    def apply(self, queue_length):
        """Apply rule to input queue length."""
        return self.queue_level.membership(queue_length) * self.output_timing

class FuzzyController:
    """Fuzzy logic controller for traffic signal timing."""
    def __init__(self):
        # Define fuzzy sets for queue length: short, medium, long
        self.short_queue = FuzzySet(0, 0, 5)
        self.medium_queue = FuzzySet(3, 7, 12)
        self.long_queue = FuzzySet(10, 15, 20)

        # Define rules: if queue is short then short timing, etc.
        self.rules = [
            FuzzyRule(self.short_queue, 10),  # Short queue -> short green time
            FuzzyRule(self.medium_queue, 20), # Medium -> medium time
            FuzzyRule(self.long_queue, 30)    # Long -> long time
        ]

    def fuzzify(self, queue_length):
        """Fuzzify the input queue length."""
        return [rule.queue_level.membership(queue_length) for rule in self.rules]

    def inference(self, memberships):
        """Apply rules to get fuzzy outputs."""
        return [m * rule.output_timing for m, rule in zip(memberships, self.rules)]

    def defuzzify(self, fuzzy_outputs):
        """Defuzzify using centroid method."""
        if sum(fuzzy_outputs) == 0:
            return 15  # Default timing
        weighted_sum = sum(fo * rule.output_timing for fo, rule in zip(fuzzy_outputs, self.rules))
        return weighted_sum / sum(fuzzy_outputs)

    def compute_timing(self, queue_lengths):
        """Compute optimal green time based on queue lengths from all directions."""
        # Average queue length for simplicity
        avg_queue = np.mean(queue_lengths)
        memberships = self.fuzzify(avg_queue)
        fuzzy_out = self.inference(memberships)
        return self.defuzzify(fuzzy_out)

# Integration example (to be called from environment)
def get_fuzzy_action(state, controller: FuzzyController):
    """Get action (green time) from fuzzy controller given state."""
    queue_lengths = state[:4]  # Assuming first 4 elements are queue lengths
    return controller.compute_timing(queue_lengths)