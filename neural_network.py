import numpy as np
from typing import List, Tuple


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [16]
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights_size = self._calculate_weights_size()

        # Estadísticas para normalización
        self.obs_mean = None
        self.obs_std = None

    def _calculate_weights_size(self) -> int:
        """Calculate total number of weights in the neural network."""
        total = 0
        prev_size = self.input_size

        for hidden_size in self.hidden_layers:
            total += prev_size * hidden_size + hidden_size  # weights + bias
            prev_size = hidden_size

        total += prev_size * self.output_size + self.output_size
        return total

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        if self.obs_mean is None:
            self.obs_mean = observation
            self.obs_std = np.ones_like(observation)
        else:
            # Update running statistics
            self.obs_mean = 0.99 * self.obs_mean + 0.01 * observation
            diff = observation - self.obs_mean
            self.obs_std = 0.99 * self.obs_std + 0.01 * np.abs(diff)

        return (observation - self.obs_mean) / (self.obs_std + 1e-8)

    def get_action(
        self,
        observation: np.ndarray,
        individual: np.ndarray
    ) -> int:
        """Compute action using the neural network."""
        # Normalize observation
        normalized_obs = self.normalize_observation(observation)

        # Extract weights and biases
        weights, biases = self._extract_weights(individual)

        # Forward pass through network
        x = normalized_obs
        for w, b in zip(weights[:-1], biases[:-1]):
            x = np.tanh(np.dot(x, w) + b)

        # Output layer with sigmoid
        output = 1 / (1 + np.exp(-(np.dot(x, weights[-1]) + biases[-1])))

        # Convert to binary action
        return 1 if output > 0.5 else 0

    def _extract_weights(
        self,
        individual: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract weights and biases from individual."""
        weights = []
        biases = []
        idx = 0

        # Hidden layers
        prev_size = self.input_size
        for hidden_size in self.hidden_layers:
            # Weights
            end = idx + prev_size * hidden_size
            w = individual[idx:end].reshape(prev_size, hidden_size)
            weights.append(w)
            idx = end

            # Bias
            end = idx + hidden_size
            b = individual[idx:end]
            biases.append(b)
            idx = end
            prev_size = hidden_size

        # Output layer
        end = idx + prev_size * self.output_size
        w = individual[idx:end].reshape(prev_size, self.output_size)
        weights.append(w)
        idx = end

        b = individual[idx:idx + self.output_size]
        biases.append(b)

        return weights, biases
