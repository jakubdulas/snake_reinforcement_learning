import pygame
import copy
from constants import BLOCK_SIZE, GREEN


class Segment(pygame.sprite.Sprite):
    def __init__(self):
        super(Segment, self).__init__()
        self.position = [-1, -1]
        self.rect = pygame.Rect(self.position[0]*BLOCK_SIZE, self.position[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        self.last_position = None
        self.color = GREEN

    def move(self, nowa_pozycja):
        self.last_position = copy.deepcopy(self.position)
        self.position = copy.deepcopy(nowa_pozycja)
        self.rect = pygame.Rect(self.position[0]*BLOCK_SIZE, self.position[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)