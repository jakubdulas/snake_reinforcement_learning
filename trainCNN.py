from SnakeBotCNN import SnakeBot
from SnakeGame import SnakeGame
import os


if __name__ == '__main__':
    playing = True
    bot = SnakeBot('weightsCNN2.h5', True)
    episodes = 100_000
    game = SnakeGame(AI_mode=True)
    score_sum = 0

    for episode in range(1, episodes+1):
        game.reset()
        playing = True
        steps_without_reward = 0

        while playing:
            # os.system('clear')
            state = game.get_state('image')
            score = game.score

            action = bot.make_move(state)

            playing = game.game_step(action)

            new_state = game.get_state('image')
            new_score = game.score
            
            reward = new_score - score

            if reward == 0:
                steps_without_reward += 1
            
            if not playing:
                reward = -1
                score_sum += game.score

            if steps_without_reward > 100:
                playing = False
                score_sum += game.score
        
                
            bot.update_table(state, action, reward, new_state, not playing)
            bot.update_epsilon(episode)


    game.quit()
