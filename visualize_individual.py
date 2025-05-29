import numpy as np
from walker2d_env import Walker2DEnv
from genetic_algorithm import GeneticAlgorithm


def main():
    # Pedir al usuario la generación a visualizar
    prompt = (
        '¿De qué generación quieres ver el mejor individuo?\n'
        '(ejemplo: 10): '
    )
    gen = int(input(prompt))
    filename = f'best_individual_gen_{gen}.npy'
    try:
        best_individual = np.load(filename)
    except FileNotFoundError:
        print(f'No se encontró el archivo {filename}.')
        print('Asegúrate de que existe.')
        return

    # Crear entorno
    env = Walker2DEnv(render_mode='human')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    hidden_size = 16  # Debe coincidir con el usado en el entrenamiento

    # Crear un dummy GA solo para usar get_action
    ga = GeneticAlgorithm(1, 1, 0.1, input_size, output_size, hidden_size)

    print(f'Visualizando el mejor individuo de la generación {gen}...')
    obs = env.reset()
    for _ in range(1000):
        action = ga.get_action(obs, best_individual)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
