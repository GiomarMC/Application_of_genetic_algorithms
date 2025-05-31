import gymnasium as gym
import time
import pygame
import numpy as np
from genetic_algorithm import GeneticAlgorithm


def train():
    # Crear entorno sin visualización para entrenamiento
    env = gym.make('CartPole-v1', render_mode=None)
    input_size = env.observation_space.shape[0]
    output_size = 1  # Una salida para decidir izquierda/derecha

    # Parámetros del AG
    population_size = 100
    generations = 50
    initial_mutation_rate = 0.2

    # Crear AG
    ga = GeneticAlgorithm(
        population_size=population_size,
        generations=generations,
        initial_mutation_rate=initial_mutation_rate,
        input_size=input_size,
        output_size=output_size
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

    # Visualizar
    state, _ = env.reset()
    total_reward = 0

    try:
        while True:
            # Obtener acción
            action = ga.nn.get_action(state, best_individual)

            # Aplicar acción
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Pequeña pausa para visualización
            time.sleep(0.016)

            # Verificar si se cierra la ventana
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Si termina el episodio, reiniciar
            if terminated or truncated:
                print(f"Episodio terminado. Recompensa total: {total_reward}")
                state, _ = env.reset()
                total_reward = 0

    except KeyboardInterrupt:
        print("\nVisualización detenida por el usuario")
    finally:
        env.close()


def main():
    # Entrenar
    train()

    # Visualizar mejor individuo
    print("\nVisualizando el mejor individuo...")
    visualize_best_individual(49)  # Última generación


if __name__ == "__main__":
    main()
