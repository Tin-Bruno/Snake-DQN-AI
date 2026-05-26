import random
import time

from envs.snake_env import SnakeEnv


def action_name(action):
    names = {
        0: "STRAIGHT",
        1: "RIGHT",
        2: "LEFT",
    }

    return names[action]


def main():
    env = SnakeEnv(render=True)

    state = env.reset()

    print("Ambiente iniciado.")
    print("Estado inicial:", state)
    print("Tamanho do estado:", len(state))
    print()

    try:
        for step in range(1000):
            action = random.randint(0, 2)

            next_state, reward, done, info = env.step(action)

            print(
                f"step={step} "
                f"action={action_name(action)} "
                f"reward={reward:.3f} "
                f"score={info['score']} "
                f"done={done}"
            )

            state = next_state
            time.sleep(0.03)

            if done:
                print("Episódio terminou.")
                print("Score final:", info["score"])
                break

    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")

    finally:
        env.close()
        print("Ambiente fechado.")


if __name__ == "__main__":
    main()