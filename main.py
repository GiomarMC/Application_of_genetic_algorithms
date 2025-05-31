import gymnasium as gym
import time
import pygame
import numpy as np
from genetic_algorithm import GeneticAlgorithm
import os
import re


def get_available_generations(outputs_dir='outputs'):
    """Get list of available generations from saved files."""
    pattern = re.compile(r'best_individual_gen_(\d+)\.npy')
    generations = []
    if os.path.exists(outputs_dir):
        for fname in os.listdir(outputs_dir):
            match = pattern.match(fname)
            if match:
                generations.append(int(match.group(1)))
    return sorted(generations)


def train():
    # Crear entorno sin visualización para entrenamiento
    env = gym.make('CartPole-v1', render_mode=None)
    input_size = env.observation_space.shape[0]
    output_size = 1  # Una salida para decidir izquierda/derecha

    # Parámetros del AG
    population_size = 100  # Aumentar tamaño de población para más diversidad
    generations = 100  # Más generaciones para mejor convergencia
    initial_mutation_rate = 0.03  # Reducir tasa de mutación inicial
    elite_size = max(5, int(0.02 * population_size))  # Aumentar elitismo
    crossover_points = 3  # Más puntos de cruce para mejor mezcla
    diversity_reintroduction_rate = 0.1  # Aumentar tasa de reintroducción
    diversity_threshold = 0.05  # Reducir umbral de diversidad
    eval_episodes = 10  # Aumentar número de episodios de evaluación
    early_stopping_patience = 20  # Más paciencia para early stopping
    min_fitness_improvement = 0.005  # Reducir mejora mínima requerida

    # Crear AG
    ga = GeneticAlgorithm(
        population_size=population_size,
        generations=generations,
        initial_mutation_rate=initial_mutation_rate,
        input_size=input_size,
        output_size=output_size,
        elite_size=elite_size,
        crossover_points=crossover_points,
        diversity_reintroduction_rate=diversity_reintroduction_rate,
        diversity_threshold=diversity_threshold,
        eval_episodes=eval_episodes,
        early_stopping_patience=early_stopping_patience,
        min_fitness_improvement=min_fitness_improvement
    )

    # Entrenar
    ga.run(env, max_steps=500)
    env.close()


def visualize_best_individual(gen: int):
    # Crear entorno con visualización
    env = gym.make('CartPole-v1', render_mode='human')
    input_size = env.observation_space.shape[0]
    output_size = 1

    # Crear AG (solo para acceder a la red neuronal)
    ga = GeneticAlgorithm(
        population_size=1,
        generations=1,
        initial_mutation_rate=0.0,
        input_size=input_size,
        output_size=output_size
    )

    # Cargar mejor individuo
    best_individual = np.load(f'outputs/best_individual_gen_{gen}.npy')

    # Cargar estadísticas si existen
    stats_file = f'outputs/stats_gen_{gen}.npy'
    if os.path.exists(stats_file):
        stats = np.load(stats_file, allow_pickle=True).item()
        best_fitness = stats['best_fitness']
        print(f"\nMejor fitness de la generación {gen}: {best_fitness:.2f}")

    print("\nPresiona 'q' para salir o cierra la ventana")
    print("El episodio continuará hasta que cierres la ventana")

    episode_count = 0
    total_reward = 0
    state, _ = env.reset()

    try:
        while True:
            # Obtener acción
            action = ga.nn.get_action(state, best_individual)

            # Aplicar acción
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Pequeña pausa para visualización
            time.sleep(0.016)

            # Verificar si se cierra la ventana o se presiona 'q'
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        raise KeyboardInterrupt

            # Si termina el episodio, reiniciar
            if terminated or truncated:
                print(
                    f"Episodio {episode_count + 1}: "
                    f"Recompensa total: {total_reward}"
                )
                state, _ = env.reset()
                total_reward = 0
                episode_count += 1

    except KeyboardInterrupt:
        print("\nVisualización detenida por el usuario")
    finally:
        env.close()


def main():
    # Entrenar
    train()

    # Obtener última generación disponible
    generations = get_available_generations()
    if not generations:
        print(
            "No se encontraron generaciones guardadas en la "
            "carpeta 'outputs'."
        )
        return

    # Visualizar mejor individuo de la última generación
    print("\nVisualizando el mejor individuo de la última generación...")
    visualize_best_individual(generations[-1])


if __name__ == "__main__":
    main()
