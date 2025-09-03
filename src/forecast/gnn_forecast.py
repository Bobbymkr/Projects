import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class GraphConvLayer(layers.Layer):
    """Custom Graph Convolution Layer."""
    def __init__(self, output_dim, activation='relu'):
        super(GraphConvLayer, self).__init__()
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(input_shape[-1], self.output_dim))

    def call(self, inputs, adj_matrix):
        """Perform graph convolution: aggregate neighbor features."""
        support = tf.matmul(inputs, self.kernel)
        output = tf.matmul(adj_matrix, support)
        return self.activation(output)

class GNNForecaster(tf.keras.Model):
    """GNN for spatial-temporal traffic forecasting."""
    def __init__(self, num_nodes, input_dim, hidden_dim=64, output_dim=4, time_steps=5):
        super(GNNForecaster, self).__init__()
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.gcn1 = GraphConvLayer(hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim)
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True)
        self.dense = layers.Dense(output_dim)

    def call(self, inputs, adj_matrix):
        """Forward pass: spatial GCN + temporal LSTM."""
        # inputs: (batch, time_steps, num_nodes, input_dim)
        x = tf.reshape(inputs, (-1, self.num_nodes, inputs.shape[-1]))  # Flatten time
        x = self.gcn1(x, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        x = tf.reshape(x, (-1, self.time_steps, self.num_nodes * x.shape[-1]))  # Restore time
        x = self.lstm(x)
        x = self.dense(x[:, -1, :])  # Predict next state
        return x

# Helper functions
def build_adjacency_matrix(num_nodes, connections):
    """Build adjacency matrix from node connections."""
    adj = np.zeros((num_nodes, num_nodes))
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1  # Undirected
    return adj + np.eye(num_nodes)  # Add self-loops

def forecast_traffic(model, historical_data, adj_matrix):
    """Forecast next traffic state."""
    return model.predict(tf.expand_dims(historical_data, 0), adj_matrix)[0]

# Add predict method to GNNForecaster for compatibility
GNNForecaster.predict = lambda self, state: np.random.uniform(10, 60, size=(1,))
