from agents.dqn_agent import DQNAgent
from config import MODEL_DIR
from envs.snake_env import SnakeEnv


def main():
    model_path = MODEL_DIR / "best_snake_dqn.pt"

    env = SnakeEnv(render=False)
    agent = DQNAgent()

    agent.load(model_path)
    agent.epsilon = 0.0

    episodes = 20
    scores = []

    print("Modelo carregado:", model_path)
    print(f"Avaliando por {episodes} episódios...")

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state

        score = info["score"]
        scores.append(score)

        print(f"Episode={episode} Score={score}")

    env.close()

    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)

    print()
    print("Resultado final:")
    print(f"Score médio: {avg_score:.2f}")
    print(f"Melhor score: {best_score}")
    print(f"Pior score: {worst_score}")


if __name__ == "__main__":
    main()