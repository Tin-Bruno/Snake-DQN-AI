import random
from collections import deque

import numpy as np
import pygame

from config import GRID_SIZE, CELL_SIZE, MAX_STEPS_WITHOUT_FOOD


class SnakeEnv:
    def __init__(self, render=False):
        self.render_enabled = render

        self.width = GRID_SIZE
        self.height = GRID_SIZE

        self.snake = None
        self.direction = None
        self.food = None
        self.score = 0
        self.steps_without_food = 0

        self.screen = None
        self.clock = None

        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * CELL_SIZE, self.height * CELL_SIZE)
            )
            pygame.display.set_caption("Snake DQN")
            self.clock = pygame.time.Clock()

    def reset(self):
        center_x = self.width // 2
        center_y = self.height // 2

        self.snake = deque([
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ])

        self.direction = (1, 0)
        self.score = 0
        self.steps_without_food = 0
        self.food = self._spawn_food()

        return self.get_state()

    def step(self, action):
        """
        Ações:
        0 = seguir reto
        1 = virar para a direita
        2 = virar para a esquerda
        """

        self.steps_without_food += 1

        self._change_direction(action)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        reward = -0.001
        done = False

        if self._is_collision(new_head):
            reward = -1.0
            done = True
            return self.get_state(), reward, done, {"score": self.score}

        self.snake.appendleft(new_head)

        if new_head == self.food:
            reward = 1.0
            self.score += 1
            self.steps_without_food = 0
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        if self.steps_without_food >= MAX_STEPS_WITHOUT_FOOD:
            reward = -0.5
            done = True

        if self.render_enabled:
            self.render()

        return self.get_state(), reward, done, {"score": self.score}

    def get_state(self):
        head = self.snake[0]

        dir_left = self.direction == (-1, 0)
        dir_right = self.direction == (1, 0)
        dir_up = self.direction == (0, -1)
        dir_down = self.direction == (0, 1)

        danger_straight = self._danger_in_direction(self.direction)
        danger_right = self._danger_in_direction(self._turn_right(self.direction))
        danger_left = self._danger_in_direction(self._turn_left(self.direction))

        food_x, food_y = self.food
        head_x, head_y = head

        state = [
            danger_straight,
            danger_right,
            danger_left,

            dir_left,
            dir_right,
            dir_up,
            dir_down,

            food_x < head_x,
            food_x > head_x,
            food_y < head_y,
            food_y > head_y,
        ]

        return np.array(state, dtype=np.float32)

    def _change_direction(self, action):
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)

    def _turn_right(self, direction):
        dx, dy = direction
        return (-dy, dx)

    def _turn_left(self, direction):
        dx, dy = direction
        return (dy, -dx)

    def _danger_in_direction(self, direction):
        head_x, head_y = self.snake[0]
        dx, dy = direction
        next_pos = (head_x + dx, head_y + dy)
        return self._is_collision(next_pos)

    def _is_collision(self, position):
        x, y = position

        if x < 0 or x >= self.width:
            return True

        if y < 0 or y >= self.height:
            return True

        if position in list(self.snake):
            return True

        return False

    def _spawn_food(self):
        while True:
            food = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )

            if food not in self.snake:
                return food

    def render(self):
        if not self.render_enabled:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((20, 20, 20))

        for x, y in self.snake:
            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(self.screen, (0, 200, 0), rect)

        food_x, food_y = self.food
        food_rect = pygame.Rect(
            food_x * CELL_SIZE,
            food_y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(self.screen, (220, 0, 0), food_rect)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_enabled:
            pygame.quit()
