from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np


# ----------------- Utilities -----------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


class Adam:
    """Minimal Adam optimizer for lists of parameter arrays."""
    def __init__(self, params: List[np.ndarray], lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.ms = [np.zeros_like(p) for p in params]
        self.vs = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]):
        self.t += 1
        b1t = self.b1 ** self.t
        b2t = self.b2 ** self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.ms[i] = self.b1 * self.ms[i] + (1 - self.b1) * g
            self.vs[i] = self.b2 * self.vs[i] + (1 - self.b2) * (g * g)
            m_hat = self.ms[i] / (1 - b1t)
            v_hat = self.vs[i] / (1 - b2t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ----------------- Q-Network (NumPy) -----------------
class QNet:
    """
    Simple fully-connected network mapping state vector to action values.
    Implemented in NumPy with two hidden layers and ReLU activations.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He initialization
        self.W1 = rng.standard_normal((state_dim, hidden), dtype=np.float32) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, hidden), dtype=np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros((hidden,), dtype=np.float32)
        self.W3 = rng.standard_normal((hidden, action_dim), dtype=np.float32) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros((action_dim,), dtype=np.float32)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass; returns output and cache for backprop."""
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        q = a2 @ self.W3 + self.b3
        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return q, cache

    def backward(self, cache: Dict[str, np.ndarray], dq: np.ndarray) -> List[np.ndarray]:
        """Backprop from dq (dLoss/dQ) to parameter grads in same order as params."""
        a2 = cache["a2"]; a1 = cache["a1"]; x = cache["x"]
        # Layer 3
        dW3 = a2.T @ dq
        db3 = dq.sum(axis=0)
        da2 = dq @ self.W3.T
        # Layer 2
        dz2 = da2 * relu_grad(cache["z2"])
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        # Layer 1
        dz1 = da1 * relu_grad(cache["z1"])
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def copy_from(self, other: "QNet"):
        for p, q in zip(self.params, other.params):
            np.copyto(p, q)

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load(self, path: str):
        data = np.load(path)
        self.W1[:] = data["W1"]; self.b1[:] = data["b1"]
        self.W2[:] = data["W2"]; self.b2[:] = data["b2"]
        self.W3[:] = data["W3"]; self.b3[:] = data["b3"]


# ----------------- Replay Buffer -----------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, s, a, r, ns, done):
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = ns
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int):
        max_i = self.size()
        idx = np.random.choice(max_i, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 20000
    batch_size: int = 64
    target_update: int = 1000
    buffer_size: int = 50000
    warmup: int = 1000
    seed: int | None = 42


class DQNAgent:
    """NumPy-based DQN agent with target network and replay buffer."""
    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.q = QNet(state_dim, action_dim, hidden=128, seed=cfg.seed)
        self.target = QNet(state_dim, action_dim, hidden=128, seed=(cfg.seed or 0) + 1)
        self.target.copy_from(self.q)
        self.buffer = ReplayBuffer(cfg.buffer_size, state_dim)
        self.steps = 0
        self.optimizer = Adam(self.q.params, lr=cfg.lr)
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        eps = 0.0 if evaluate else self._epsilon()
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.action_dim))
        q, _ = self.q.forward(state.reshape(1, -1))
        return int(np.argmax(q[0]))

    def _epsilon(self) -> float:
        eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * math.exp(-1.0 * self.steps / self.cfg.eps_decay)
        return float(eps)

    def push(self, s, a, r, ns, done):
        self.buffer.add(s, a, r, ns, float(done))

    def train_step(self):
        if self.buffer.size() < max(self.cfg.warmup, self.cfg.batch_size):
            self.steps += 1
            return None
        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)

        # Compute targets using target network
        q_next, _ = self.target.forward(ns)
        max_next = np.max(q_next, axis=1)
        target = r + self.cfg.gamma * (1.0 - d) * max_next

        # Forward current network and gather Q(s,a)
        q, cache = self.q.forward(s)
        # Build gradient wrt q outputs
        dq = np.zeros_like(q)
        batch_indices = np.arange(self.cfg.batch_size)
        dq[batch_indices, a] = (q[batch_indices, a] - target) * (2.0 / self.cfg.batch_size)

        grads = self.q.backward(cache, dq)
        self.optimizer.step(grads)

        if self.steps % self.cfg.target_update == 0:
            self.target.copy_from(self.q)
        self.steps += 1
        # Return loss estimate (MSE)
        loss = np.mean((q[batch_indices, a] - target) ** 2)
        return float(loss)

    def save(self, path: str):
        self.q.save(path)

    def load(self, path: str):
        self.q.load(path)
        self.target.copy_from(self.q)
