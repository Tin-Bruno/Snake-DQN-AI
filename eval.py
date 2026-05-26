import time

from agents.dqn_agent import DQNAgent
from config import MODEL_DIR
from envs.snake_env import SnakeEnv


def main():
    model_path = MODEL_DIR / "best_snake_dqn.pt"

    env = SnakeEnv(render=True)
    agent = DQNAgent()

    agent.load(model_path)
    agent.epsilon = 0.0

    state = env.reset()
    done = False

    print("Modelo carregado:", model_path)
    print("Avaliando com epsilon = 0.0")

    try:
        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            state = next_state

            time.sleep(0.03)

        print("Score final:", info["score"])

    except KeyboardInterrupt:
        print("Avaliação interrompida pelo usuário.")

    finally:
        env.close()


if __name__ == "__main__":
    main()