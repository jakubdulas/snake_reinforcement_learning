from SnakeBotDNN import SnakeBot
from SnakeGame import SnakeGame
import pygame


if __name__ == '__main__':
    # bot = SnakeBot('weightsDNN_without_segments.h5')
    game = SnakeGame(add_segments=True, AI_mode=False)
    state_type = 'vector'

    playing = True

    while playing:
        state = game.get_state(state_type)
        score = game.score

        # action = bot.make_move(state)
        playing = game.game_step()
    
        game.add_text(f"Max score: {game.max_score}", (30, 15))
        game.add_text(f"Score: {game.score}", (30, 40))

        # pygame.time.wait(200)

    game.quit()