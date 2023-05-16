import pygame
import random
from constants import BLOCK_SIZE, GRID_SIZE, RED


class Apple(pygame.sprite.Sprite):
    def __init__(self, snake):
        super(Apple, self).__init__()
        possible_positions = []
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                if snake.position == (i, j):
                    continue

                for segment in snake.segments:
                    if segment.position == (i, j):
                        break
                else:
                    possible_positions.append((i, j))

        self.color = RED
        random_position = random.choice(possible_positions)
        self.position = random_position

        self.rect = pygame.Rect((random_position[0])*BLOCK_SIZE, (random_position[1])*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)