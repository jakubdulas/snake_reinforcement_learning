from Apple import Apple
from Snake import Snake
from pygame.sprite import Group
import pygame
from constants import *
from Direction import Direction
import cv2
import numpy as np
from Queue import Queue

class SnakeGame:
    def __init__(self, add_segments=True, AI_mode=False) -> None:
        pygame.init()

        self.texts = []
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Comic Sans MS', 20)
        self.max_score = 0
        self.PORUSZ_WEZEM = pygame.USEREVENT + 1
        self.add_segments = add_segments
        self.AI_mode = AI_mode
        pygame.time.set_timer(self.PORUSZ_WEZEM, 200)
        self.frames = Queue(2)

        self.reset_frames()

        for _ in range(2):
            img = np.zeros((1, 64, 64, 1))
            self.frames.add(img)
        
        self.reset()

    def reset_frames(self):
        for _ in range(2):
            img = np.zeros((1, 64, 64, 1))
            self.frames.add(img)

    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake)
        self.score = 0
        self.is_lasting = True
        self.reset_frames()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_lasting = False
                if event.key == pygame.K_w:
                    self.snake.change_direction(Direction.UP)
                if event.key == pygame.K_s:
                    self.snake.change_direction(Direction.DOWN)
                if event.key == pygame.K_a:
                    self.snake.change_direction(Direction.LEFT)
                if event.key == pygame.K_d:
                    self.snake.change_direction(Direction.RIGHT)
            elif event.type == pygame.QUIT:
                self.is_lasting = False
                break
            elif event.type == self.PORUSZ_WEZEM:
                self.snake.update()

    def display_background(self):
        self.screen.fill(BLACK)

    def display_objects(self):
        if self.add_segments:
            self.snake.draw_segments(self.screen)

        pygame.draw.rect(self.screen, self.snake.color, self.snake.rect)
        pygame.draw.rect(self.screen, self.apple.color, self.apple.rect)

    def add_text(self, text, position, color=WHITE):
        rendered_text = self.font.render(text,  False, color)
        self.texts.append((rendered_text, position))

    def display_text(self):
        for text, position in self.texts:
            self.screen.blit(text, position)
        self.texts = []
        
    def is_colliding(self):
        if self.snake.check_collision():
            return True
        return False

    def get_state(self, type='vector'):
        if type == 'image':
            return self.frames.array
        
        if type == 'map':
            map = np.zeros(GRID_SIZE)
            x, y = self.snake.position
            map[y, x] = 1
            for segment in self.snake.segments:
                x, y = segment.position
                map[y, x] = 2
            x, y = self.apple.position
            map[y, x] = 3
            return map
        
        ax = self.apple.position[0]
        ay = self.apple.position[1]
        sx = self.snake.position[0]
        sy = self.snake.position[1]

        d_wall_snake_x = GRID_SIZE[0]-1-sx
        d_wall_snake_y = GRID_SIZE[1]-1-sy
        
        if not self.add_segments:
            return (
                int(self.snake.direction == Direction.UP),
                int(self.snake.direction == Direction.RIGHT),
                int(self.snake.direction == Direction.DOWN),
                int(self.snake.direction == Direction.LEFT),

                int(ax == sx),
                int(ay == sy),
                int(ax > sx), 
                int(ay > sy),

                int(sy == 0),
                int(d_wall_snake_x == 0),
                int(d_wall_snake_y == 0),
                int(sx == 0),
            )
        
        up = False
        right = False
        down = False
        left = False
    

        if len(self.snake.segments):
            lsx = self.snake.segments[-1].position[0]
            lsy = self.snake.segments[-1].position[1]
        else:
            lsx = -1
            lsy = -1
        

        for segment in self.snake.segments:
            pos = segment.position
            if self.snake.position[0] == pos[0]:
                if self.snake.position[1] - 1== pos[1]:
                    up = True
                if self.snake.position[1] + 1== pos[1]:
                    down = True
            if self.snake.position[1] == pos[1]:
                if self.snake.position[0] - 1 == pos[0]:
                    left = True
                if self.snake.position[0] + 1 == pos[0]:
                    right = True

        return (
            int(self.snake.direction == Direction.UP),
            int(self.snake.direction == Direction.RIGHT),
            int(self.snake.direction == Direction.DOWN),
            int(self.snake.direction == Direction.LEFT),

            int(ax == sx),
            int(ay == sy),
            int(ax > sx), 
            int(ay > sy),

            int(up or (sy == 0)),
            int(right or (d_wall_snake_x == 0)),
            int(down or (d_wall_snake_y == 0)),
            int(left or (sx == 0)),

            int(len(self.snake.segments) > 0),
            int(lsx == sx),
            int(lsy == sy),
            int(lsx > sx), 
            int(lsy > sy),
        )
        
    
    def score_a_point(self):
        self.apple = Apple(self.snake)
        self.score += 1

        if self.add_segments:
            self.snake.eat()

        if self.score > self.max_score:
            self.max_score = self.score

    def take_action(self, action):
        self.snake.change_direction(action)

    def quit(self):
        pygame.quit()

    def save_frame(self):
        img = np.array(pygame.surfarray.array3d(self.screen))
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (64, 64, 1))
        img = np.expand_dims(img, axis=0)
        self.frames.add(img)

    def game_step(self, action=None):
        if action is not None:
            self.take_action(action)

        if not self.AI_mode:
            self.handle_events()
        else:
            self.snake.update()

        self.display_background()

        if self.apple.position == self.snake.position:
            self.score_a_point()
        
        self.display_objects()
        self.display_text()
        self.save_frame()

        self.clock.tick(30)
        pygame.display.flip()
        
        if self.is_colliding():
            self.frames.add(np.zeros((1, 64, 64, 1)))
            self.is_lasting = False
        
        return self.is_lasting
    

if __name__ == '__main__':
    playing = True
    game = SnakeGame()

    while playing:
        playing = game.game_step()

        print(game.get_state('map'))

        game.add_text(f"Max score: {game.max_score}", (30, 35))
        game.add_text(f"Score: {game.score}", (30, 60))

    game.quit()