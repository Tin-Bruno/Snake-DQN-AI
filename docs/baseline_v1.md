# Baseline v1 - Snake DQN

## Objetivo

Registrar o primeiro modelo funcional do projeto Snake-DQN-AI usando DQN simples.

Esta baseline serve como comparação antes de alterar reward, arquitetura, treino ou estado da IA.

## Estrutura atual

- Ambiente: Snake customizado com Pygame
- Algoritmo: DQN
- Estado: 11 valores
- Ações: 3 ações
  - 0 = seguir reto
  - 1 = virar direita
  - 2 = virar esquerda
- Replay Buffer: sim
- Target Network: sim
- Epsilon-greedy: sim

## Estado usado pela IA

O estado possui 11 valores:

1. perigo à frente
2. perigo à direita
3. perigo à esquerda
4. direção esquerda
5. direção direita
6. direção cima
7. direção baixo
8. comida está à esquerda
9. comida está à direita
10. comida está acima
11. comida está abaixo

## Reward atual

```text
+1.0   pegou comida
-1.0   morreu
-0.5   ficou muito tempo sem pegar comida
-0.001 por passo