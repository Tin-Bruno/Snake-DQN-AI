import numpy as np

from memory.replay_buffer import ReplayBuffer


def main():
    buffer = ReplayBuffer(capacity=5)

    for i in range(10):
        state = np.array([i], dtype=np.float32)
        action = i % 3
        reward = float(i)
        next_state = np.array([i + 1], dtype=np.float32)
        done = i % 2 == 0

        buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    print("Tamanho do buffer:", len(buffer))

    batch = buffer.sample(batch_size=3)

    print("\nAmostra do buffer:")
    for experience in batch:
        state, action, reward, next_state, done = experience

        print(
            "state:", state,
            "| action:", action,
            "| reward:", reward,
            "| next_state:", next_state,
            "| done:", done,
        )


if __name__ == "__main__":
    main()