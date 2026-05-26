import torch

from models.dqn import DQN


def main():
    model = DQN(input_size=11, output_size=3)

    fake_state = torch.zeros((1, 11), dtype=torch.float32)

    q_values = model(fake_state)

    print("Entrada:", fake_state)
    print("Saída Q-values:", q_values)
    print("Formato da saída:", q_values.shape)


if __name__ == "__main__":
    main()