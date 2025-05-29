import numpy as np
import matplotlib.pyplot as plt
import os


class GeneticAlgorithm:
    def __init__(
        self, population_size, generations, mutation_rate,
        input_size, output_size, hidden_size=16
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # Total de parámetros: W1 + b1 + W2 + b2
        self.weights_size = (
            input_size * hidden_size + hidden_size +
            hidden_size * output_size + output_size
        )
        self.population = self.initialize_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []

        # Crear directorio de outputs si no existe
        self.output_dir = 'outputs'
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_population(self):
        # Inicializar pesos cerca de 0 (normal)
        return np.random.normal(
            0, 0.1, (self.population_size, self.weights_size)
        )

    def get_action(self, observation, individual):
        # Extraer pesos y bias
        idx = 0
        # Capa 1
        end1 = idx + self.input_size * self.hidden_size
        W1 = individual[idx:end1].reshape(self.input_size, self.hidden_size)
        idx = end1
        b1 = individual[idx:idx + self.hidden_size]
        idx += self.hidden_size
        # Capa 2
        end2 = idx + self.hidden_size * self.output_size
        W2 = individual[idx:end2].reshape(self.hidden_size, self.output_size)
        idx = end2
        b2 = individual[idx:idx + self.output_size]
        # Forward pass
        h = np.tanh(np.dot(observation, W1) + b1)
        action = np.dot(h, W2) + b2
        return action

    def evaluate_fitness(self, individual, env, max_steps=1000):
        observation = env.reset()
        initial_x = observation[0]
        survived_steps = 0
        retroceso_penalty = 0
        stable_height_bonus = 0
        angle_penalty = 0
        for step in range(max_steps):
            action = self.get_action(observation, individual)
            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high
            )
            observation, reward, terminated, truncated, info = env.step(action)
            current_x = observation[0]
            current_height = observation[1]
            if current_x < initial_x:
                retroceso_penalty += abs(current_x - initial_x)
            if 0.9 < current_height < 1.4:
                stable_height_bonus += 0.5
            angle_penalty += 0.1 * abs(observation[2])  # penalización suave
            survived_steps += 1
            if terminated or truncated:
                break
        avance_x = observation[0] - initial_x
        fitness = (
            avance_x + survived_steps - 0.2 * retroceso_penalty +
            stable_height_bonus - angle_penalty
        )
        return fitness

    def select_parents(self, fitness_scores):
        # Selección por torneo
        parents = []
        for _ in range(self.population_size):
            idxs = np.random.choice(len(fitness_scores), 3, replace=False)
            winner = idxs[np.argmax(fitness_scores[idxs])]
            parents.append(self.population[winner])
        return np.array(parents)

    def crossover(self, parents):
        # Cruce de punto único
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                point = np.random.randint(1, self.weights_size)
                child1 = np.concatenate([
                    parents[i][:point],
                    parents[i+1][point:]
                ])
                child2 = np.concatenate([
                    parents[i+1][:point],
                    parents[i][point:]
                ])
                offspring.extend([child1, child2])
        return np.array(offspring)

    def mutate(self, offspring):
        # Mutación gaussiana
        mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
        mutation = np.random.normal(0, 0.1, offspring.shape)
        offspring[mutation_mask] += mutation[mutation_mask]
        return offspring

    def plot_learning_curve(self):
        """Genera y guarda la gráfica de la curva de aprendizaje."""
        plt.figure(figsize=(10, 6))
        generations = range(1, len(self.best_fitness_history) + 1)

        plt.plot(
            generations,
            self.best_fitness_history,
            'b-',
            label='Mejor Fitness'
        )
        plt.plot(
            generations,
            self.avg_fitness_history,
            'r-',
            label='Fitness Promedio'
        )

        plt.title('Curva de Aprendizaje del AG')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)

        # Guardar la gráfica en el directorio de outputs
        plt.savefig(os.path.join(self.output_dir, 'learning_curve.png'))
        plt.close()

    def run(self, env, max_steps=1000):
        for gen in range(self.generations):
            fitness_scores = np.array([
                self.evaluate_fitness(ind, env, max_steps)
                for ind in self.population
            ])
            best_fitness = np.max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            print(
                f"Gen {gen+1}/{self.generations} | "
                f"Best: {best_fitness:.2f} | "
                f"Avg: {avg_fitness:.2f}"
            )
            # Guardar el mejor individuo de la generación
            # en el directorio de outputs
            best_individual = self.population[np.argmax(fitness_scores)]
            np.save(
                os.path.join(
                    self.output_dir,
                    f'best_individual_gen_{gen+1}.npy'
                ),
                best_individual
            )
            parents = self.select_parents(fitness_scores)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            # Elitismo: mantener al mejor
            elite_idx = np.argmax(fitness_scores)
            offspring[0] = self.population[elite_idx]
            self.population = offspring
        print("\nEntrenamiento completado.")
        self.plot_learning_curve()
