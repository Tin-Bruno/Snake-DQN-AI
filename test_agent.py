from agents.dqn_agent import DQNAgent
from envs.snake_env import SnakeEnv


def main():
    env = SnakeEnv(render=False)
    agent = DQNAgent()

    state = env.reset()

    print("Device:", agent.device)
    print("Epsilon inicial:", agent.epsilon)
    print("Tamanho do estado:", len(state))

    for step in range(1200):
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

        state = next_state

        if done:
            state = env.reset()

        if step % 100 == 0:
            print(
                f"step={step} "
                f"memory={len(agent.memory)} "
                f"loss={loss}"
            )

    agent.update_target_network()
    agent.decay_epsilon()

    print("Epsilon após decay:", agent.epsilon)
    print("Teste do agente finalizado.")

    env.close()


if __name__ == "__main__":
    main()