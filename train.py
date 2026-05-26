from agents.dqn_agent import DQNAgent
from config import EPISODES, MODEL_DIR, TARGET_UPDATE_EVERY
from envs.snake_env import SnakeEnv


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = SnakeEnv(render=False)
    agent = DQNAgent()

    best_score = 0

    for episode in range(1, EPISODES + 1):
        state = env.reset()

        done = False
        total_reward = 0.0
        last_loss = None

        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            agent.memory.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            loss = agent.train_step()

            if loss is not None:
                last_loss = loss

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        score = info["score"]

        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()

        if score > best_score:
            best_score = score
            agent.save(MODEL_DIR / "best_snake_dqn.pt")

        agent.save(MODEL_DIR / "last_snake_dqn.pt")

        print(
            f"Episode={episode} "
            f"Score={score} "
            f"Best={best_score} "
            f"Reward={total_reward:.3f} "
            f"Epsilon={agent.epsilon:.3f} "
            f"Loss={last_loss}"
        )

    env.close()


if __name__ == "__main__":
    main()