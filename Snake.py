import pygame
from Direction import Direction
from Segment import Segment
import copy
from constants import BLOCK_SIZE, GRID_SIZE, BLUE

class Snake(pygame.sprite.Sprite):
    ACTIONS = [
        Direction.UP,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT
    ]

    def __init__(self):
        self.position = (GRID_SIZE[0]//2, GRID_SIZE[1]//2)
        self.rect = pygame.Rect(self.position[0]*BLOCK_SIZE, self.position[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        self.color = BLUE

        self.direction = Direction.UP
        self.new_direction = Direction.UP

        self.last_position = self.position
        self.add_segment = False
        self.segments = []

    
    def change_direction(self, direction: Direction) -> None:
        can_change_direction = True
        
        if abs(self.direction.value-direction.value) == 2:
            can_change_direction = False

        if can_change_direction:
            self.new_direction = direction


    def update(self):
        self.direction = self.new_direction

        self.last_position = copy.deepcopy(self.position)

        if self.direction == Direction.UP:
            self.position =  (self.position[0], self.position[1] - 1)
        if self.direction == Direction.RIGHT:
            self.position = (self.position[0] + 1, self.position[1])
        if self.direction == Direction.LEFT:
            self.position = (self.position[0] - 1, self.position[1])
        if self.direction == Direction.DOWN:
            self.position = (self.position[0], self.position[1] + 1)

        self.rect = pygame.Rect(self.position[0]*BLOCK_SIZE, self.position[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

        for i in range(len(self.segments)):
            if i == 0:
                self.segments[i].move(self.last_position)
            else:
                self.segments[i].move(self.segments[i-1].last_position)

        if self.add_segment:
            new_segment = Segment()
            new_position = None

            if len(self.segments) > 0:
                new_position = copy.deepcopy(self.segments[-1].position)
            else:
                new_position = copy.deepcopy(self.last_position)
                new_segment.position = new_position

            self.segments.append(new_segment)
            self.add_segment = False
            
    def draw_segments(self, screen):
        for segment in self.segments:
            pygame.draw.rect(screen, segment.color, segment.rect)
    
    def eat(self):
        self.add_segment = True

    def check_collision(self):
        for segment in self.segments:
            if self.position == segment.position:
                return True

        if self.position[1] < 0 or self.position[1] >= GRID_SIZE[1]:
            return True
        if self.position[0] < 0 or self.position[0] >= GRID_SIZE[0]:
            return True
        
        return False 
    
    def get_possible_actions(self):
        possible_actions = []
        for direction in self.ACTIONS:
            if abs(self.direction.value-direction.value) != 2:
                possible_actions.append(direction)

        return possible_actions
