import time

from agents.dqn_agent import DQNAgent
from envs.snake_env import SnakeEnv
from config import MODEL_DIR


def main():
    model_path = MODEL_DIR / "best_snake_dqn.pt"

    env = SnakeEnv(render=True)
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.0

    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        time.sleep(0.03)

    print(f"Score final: {info['score']}")

    env.close()


if __name__ == "__main__":
    main()
