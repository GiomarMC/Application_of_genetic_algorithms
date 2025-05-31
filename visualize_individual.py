import gymnasium as gym
import time
import pygame
import numpy as np
import os
import re
from genetic_algorithm import GeneticAlgorithm


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


def visualize_best_individual(gen: int):
    """Visualize the best individual from a specific generation."""
    # Crear entorno con visualización
    env = gym.make('CartPole-v1', render_mode='human')
    input_size = env.observation_space.shape[0]
    output_size = 1  # Una salida para decidir izquierda/derecha

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
    state, _ = env.reset()
    total_reward = 0
    episode_count = 0
    max_episodes = 5  # Número de episodios a visualizar

    try:
        while episode_count < max_episodes:
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


if __name__ == "__main__":
    # Obtener generaciones disponibles
    generations = get_available_generations()
    if not generations:
        print(
            "No se encontraron generaciones guardadas en la "
            "carpeta 'outputs'."
        )
        exit(1)

    # Mostrar información de generaciones disponibles
    print(f"\nGeneraciones disponibles: {generations[0]} a {generations[-1]}")
    print("Cada generación se visualizará durante 5 episodios.")
    print("Presiona Ctrl+C para detener la visualización.\n")

    # Selección de generación
    while True:
        try:
            prompt = (
                f"¿Qué generación quieres visualizar? "
                f"({generations[0]}-{generations[-1]}): "
            )
            gen = int(input(prompt))
            if gen in generations:
                break
            else:
                print(f"Por favor, elige una generación entre "
                      f"{generations[0]} y {generations[-1]}.")
        except ValueError:
            print("Por favor, ingresa un número válido.")

    # Visualizar la generación seleccionada
    visualize_best_individual(gen)
