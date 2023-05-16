from SnakeBotDNN import SnakeBot
from SnakeGame import SnakeGame


if __name__ == '__main__':
    playing = True
    bot = SnakeBot('weightsDNN_without_segments.h5', True)
    episodes = 10_000
    game = SnakeGame(add_segments=False, AI_mode=True)
    score_sum = 0
    start_episode = 1

    for episode in range(start_episode, episodes+1):
        game.reset()
        playing = True
        steps_without_reward = 0

        while playing:
            # os.system('clear')
            state = game.get_state()
            score = game.score

            action = bot.make_move(state)

            playing = game.game_step(action)

            new_state = game.get_state()
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


            game.add_text(f"Mean score: {score_sum/episode}", (30, 10))
            game.add_text(f"Max score: {game.max_score}", (30, 35))
            game.add_text(f"Score: {game.score}", (30, 60))
            game.add_text(f"Episode: {episode}", (30, 85))
            game.add_text(f"States: {len(bot.q_table.keys())}", (30, 110))
            game.add_text(f"Epsilon: {bot.epsilon}", (30, 135))

            # pygame.time.wait(1)

    game.quit()
