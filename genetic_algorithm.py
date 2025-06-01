import os
from typing import List
import multiprocessing as mp
import signal
from neural_network import NeuralNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import sys


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        generations: int,
        initial_mutation_rate: float,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [16],
        elite_size: int = 1,
        crossover_points: int = 1,
        diversity_threshold: float = 0.1,
        diversity_reintroduction_rate: float = 0.1,
        eval_episodes: int = 5,
        early_stopping_patience: int = 10,
        min_fitness_improvement: float = 0.01
    ):
        # Mostrar información de GPU
        print("\n=== Información de GPU ===")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Número de GPUs: {torch.cuda.device_count()}")
            print(f"GPU actual: {torch.cuda.get_device_name(0)}")
            print(f"Versión de CUDA: {torch.version.cuda}")
        print("========================\n")

        self.population_size = population_size
        self.generations = generations
        self.initial_mutation_rate = initial_mutation_rate
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.elite_size = elite_size
        self.crossover_points = crossover_points
        self.diversity_threshold = diversity_threshold
        self.diversity_reintroduction_rate = diversity_reintroduction_rate
        self.should_stop = False
        self.eval_episodes = eval_episodes
        self.early_stopping_patience = early_stopping_patience
        self.min_fitness_improvement = min_fitness_improvement
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        # Crear red neuronal
        self.nn = NeuralNetwork(input_size, output_size, hidden_layers)
        # Initialize population and history
        self.population = self.initialize_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.fitness_scores_history = []
        self.diversity_history = []
        # Create output directory
        self.output_dir = 'outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        # Detectar y configurar dispositivo (GPU/CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Usando dispositivo: {self.device}")
        # Configurar paralelización
        if self.device.type == 'cuda':
            self.use_gpu = True
            self.num_workers = torch.cuda.device_count()
        else:
            self.use_gpu = False
            self.num_workers = mp.cpu_count()
        print(f"Usando {self.num_workers} workers para paralelización")

    def _signal_handler(self, signum, frame):
        """Manejador de señal para detener el entrenamiento."""
        print("\nDeteniendo el entrenamiento...")
        self.should_stop = True
        if hasattr(self, 'pool'):
            self.pool.terminate()
            self.pool.join()
        sys.exit(0)

    def initialize_population(self) -> np.ndarray:
        """Initialize population with weights near zero."""
        return np.random.normal(
            0, 0.1,
            (self.population_size, self.nn.weights_size)
        )

    def calculate_fitness(
        self,
        steps: int,
        mean_angle: float,
        reached_max_steps: bool
    ) -> float:
        """Calculate fitness based on steps and reward for angle close to 90
        degrees."""
        # Recompensa base por pasos
        fitness = steps

        # Recompensa adicional si el ángulo promedio está cerca de 90 grados
        ideal_angle = np.pi/2
        angle_diff = abs(mean_angle - ideal_angle)
        if angle_diff < 0.1:  # Muy cerca de 90 grados
            fitness += 120
        elif angle_diff < 0.2:
            fitness += 80
        elif angle_diff < 0.3:
            fitness += 40

        # Bonus por llegar a max_steps
        if reached_max_steps:
            fitness += 100
        return max(0, fitness)

    def evaluate_individual(
        self,
        individual: np.ndarray,
        env,
        max_steps: int
    ) -> float:
        """Evaluate individual's fitness using the average of multiple
        episodes."""
        rewards = []
        for _ in range(self.eval_episodes):
            state, _ = env.reset()
            steps = 0
            angles = []  # Lista para guardar todos los ángulos

            for _ in range(max_steps):
                action = self.nn.get_action(state, individual)
                state, _, terminated, truncated, _ = env.step(action)

                # Guardar el ángulo
                pole_angle = state[2]
                angles.append(pole_angle)
                steps += 1
                if terminated or truncated:
                    break

            # Calcular el ángulo promedio
            angles = np.array(angles)
            mean_angle = np.mean(angles)

            # Calcular fitness para este episodio
            episode_fitness = self.calculate_fitness(
                steps=steps,
                mean_angle=mean_angle,
                reached_max_steps=(steps == max_steps)
            )
            rewards.append(episode_fitness)
        # Usar el promedio de los puntajes
        return np.mean(rewards)

    def get_mutation_rate(self, generation: int) -> float:
        """Get mutation rate for current generation."""
        return self.initial_mutation_rate * (1 - generation / self.generations)

    def select_parents(self, fitness_scores: np.ndarray) -> np.ndarray:
        """Seleccionar padres usando selección por torneo de tamaño 2
        (en vez de ruleta)."""
        parents = []
        tournament_size = 2
        for _ in range(len(fitness_scores) - self.elite_size):
            # Selección por torneo
            indices = np.random.choice(
                len(fitness_scores), tournament_size, replace=False
            )
            tournament_fitness = fitness_scores[indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        return np.array(parents)

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """Perform crossover between parents."""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                p1, p2 = parents[i], parents[i + 1]
                point = np.random.randint(0, self.nn.weights_size)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
                offspring.extend([c1, c2])
            else:
                offspring.append(parents[i])
        return np.array(offspring)

    def mutate(
        self, offspring: np.ndarray, generation: int,
        mutation_rate: float = None
    ) -> np.ndarray:
        """Mutar descendencia con una tasa mínima de mutación."""
        min_mutation_rate = 0.01  # Tasa mínima de mutación
        if mutation_rate is None:
            mutation_rate = self.get_mutation_rate(generation)
        mutation_rate = max(mutation_rate, min_mutation_rate)  # Aplicar mínimo
        mask = np.random.random(offspring.shape) < mutation_rate
        mutation = np.random.normal(0, 0.1, offspring.shape)
        offspring[mask] += mutation[mask]
        return offspring

    def maintain_diversity(self, fitness_scores: np.ndarray) -> None:
        """Maintain population diversity."""
        if len(self.best_fitness_history) > 1:
            if (self.best_fitness_history[-1] - self.best_fitness_history[-2]
                    < self.diversity_threshold):
                # Reintroduce random individuals
                n_replace = min(
                    int(len(fitness_scores) *
                        self.diversity_reintroduction_rate),
                    len(fitness_scores)
                )
                replace_indices = np.random.choice(
                    len(fitness_scores),
                    size=n_replace,
                    replace=False
                )
                new_individuals = self.initialize_population()[:n_replace]
                self.population[replace_indices] = new_individuals

    def calculate_population_diversity(self) -> float:
        """Calculate population diversity using average pairwise distance."""
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(
                    self.population[i] - self.population[j]
                )
                distances.append(distance)
        return np.mean(distances) if distances else 0.0

    def normalize_fitness(self, fitness_scores: np.ndarray) -> np.ndarray:
        """Normalize fitness scores to [0, 1] range."""
        min_fitness = np.min(fitness_scores)
        max_fitness = np.max(fitness_scores)
        if max_fitness == min_fitness:
            return np.ones_like(fitness_scores)
        return (fitness_scores - min_fitness) / (max_fitness - min_fitness)

    def plot_learning_curve(self) -> None:
        """Plot learning curve with additional metrics."""
        plt.figure(figsize=(12, 8))

        # Plot fitness metrics
        plt.subplot(2, 1, 1)
        plt.plot(self.best_fitness_history, label='Mejor Fitness')
        plt.plot(self.avg_fitness_history, label='Fitness Promedio')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Curva de Aprendizaje')
        plt.legend()
        plt.grid(True)

        # Plot diversity
        plt.subplot(2, 1, 2)
        plt.plot(self.diversity_history, label='Diversidad de Población')
        plt.xlabel('Generación')
        plt.ylabel('Diversidad')
        plt.title('Evolución de la Diversidad')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_curve.png'))
        plt.close()

    def run(self, env, max_steps: int = 500) -> None:
        """Ejecutar el algoritmo genético con mejoras de diversidad."""
        try:
            for gen in range(self.generations):
                if self.should_stop:
                    break

                if self.use_gpu:
                    # Evaluación en GPU
                    with torch.cuda.device(0):
                        population_tensor = torch.tensor(
                            self.population, device=self.device
                        )
                        fitness_scores = []
                        for i in range(
                            0, len(population_tensor), self.num_workers
                        ):
                            batch = population_tensor[i:i + self.num_workers]
                            batch_scores = []
                            for individual in batch:
                                score = self.evaluate_individual(
                                    individual.cpu().numpy(), env, max_steps
                                )
                                batch_scores.append(score)
                            fitness_scores.extend(batch_scores)
                        fitness_scores = np.array(fitness_scores)
                else:
                    with mp.Pool(processes=self.num_workers) as pool:
                        evaluate_func = partial(
                            self.evaluate_individual,
                            env=env,
                            max_steps=max_steps
                        )
                        fitness_scores = np.array(
                            pool.map(evaluate_func, self.population)
                        )

                diversity = self.calculate_population_diversity()
                self.diversity_history.append(diversity)
                normalized_fitness = self.normalize_fitness(fitness_scores)
                best_fitness = np.max(fitness_scores)
                avg_fitness = np.mean(fitness_scores)
                self.best_fitness_history.append(best_fitness)
                self.avg_fitness_history.append(avg_fitness)
                self.fitness_scores_history.append(fitness_scores)

                print(
                    f"Gen {gen+1}/{self.generations} | "
                    f"Best: {best_fitness:.2f} | "
                    f"Avg: {avg_fitness:.2f} | "
                    f"Diversity: {diversity:.4f}"
                )

                # Selección y reproducción con fitness normalizado
                parents = self.select_parents(normalized_fitness)
                offspring = self.crossover(parents)

                # Ensure offspring size matches population size
                if len(offspring) < self.population_size:
                    # If we have fewer offspring than needed, duplicate some
                    n_missing = self.population_size - len(offspring)
                    indices = np.random.randint(
                        0, len(offspring), size=n_missing
                    )
                    extra_offspring = offspring[indices]
                    offspring = np.concatenate([offspring, extra_offspring])
                elif len(offspring) > self.population_size:
                    # If we have too many offspring, trim the excess
                    offspring = offspring[:self.population_size]

                # Mutación con tasa mínima
                mutation_rate = self.initial_mutation_rate
                offspring = self.mutate(offspring, gen, mutation_rate)

                # Elitismo
                elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
                elite = self.population[elite_indices]
                if self.elite_size <= len(offspring):
                    offspring[:self.elite_size] = elite
                else:
                    offspring[:len(offspring)] = elite[:len(offspring)]

                if (gen + 1) % 10 == 0:
                    n_inject = max(1, int(0.05 * self.population_size))
                    inject_indices = np.random.choice(
                        len(offspring), n_inject, replace=False
                    )
                    new_individuals = self.initialize_population()[:n_inject]
                    offspring[inject_indices] = new_individuals
                self.maintain_diversity(fitness_scores)
                self.population = offspring

                # Guardar mejor individuo de la generación
                best_idx = np.argmax(fitness_scores)
                filename = f'best_individual_gen_{gen}.npy'
                np.save(
                    os.path.join(self.output_dir, filename),
                    self.population[best_idx]
                )

                # Guardar estadísticas de la generación
                stats = {
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'diversity': diversity,
                    'fitness_scores': fitness_scores
                }
                np.save(
                    os.path.join(self.output_dir, f'stats_gen_{gen}.npy'),
                    stats
                )

        except KeyboardInterrupt:
            print("\nDeteniendo el entrenamiento...")
            self.should_stop = True
        finally:
            self.plot_learning_curve()
            print("\nEntrenamiento completado.")
