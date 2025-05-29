from walker2d_env import Walker2DEnv
from genetic_algorithm import GeneticAlgorithm


def main():
    # Crear entorno
    env = Walker2DEnv(render_mode='human')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    # Parámetros del AG
    population_size = 200
    generations = 100
    mutation_rate = 0.2

    # Crear AG
    ga = GeneticAlgorithm(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        input_size=input_size,
        output_size=output_size
    )

    # Entrenar
    ga.run(env, max_steps=1000)

    # Visualizar el mejor individuo
    print("\nMostrando el mejor individuo encontrado...")
    best_idx = ga.best_fitness_history.index(max(ga.best_fitness_history))
    best_ind = ga.population[best_idx]
    obs = env.reset()
    for _ in range(1000):
        action = ga.get_action(obs, best_ind)
        action = action.clip(env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
