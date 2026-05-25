"""
Snake Eater
Made with PyGame
"""
import ast
import json
import math
import os
from operator import truediv
from collections import defaultdict

import pygame, sys, time, random, heapq
import matplotlib.pyplot as plt

#this changes the tick speed which changes how fast the snake moves
# Difficulty settings
# Easy      ->  10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
# Training Speed -> 2**63
difficulty = 40

# Window size
# Small     ->  360x240
# Medium    ->  540x380
# Large     ->  720x480
# Very Large->  1080x720
frame_size_x = 360
frame_size_y = 240

#Mode Selection
game_mode = 1 # 1 = normal; 2 = 2 snake training
mode_selected = False

# Q-Learning reward structure
got_food = 10
death = -50
move_no_food = -1
food_distance = 0.5
spiral_penalty = -10
max_steps_without_food = (frame_size_x//10) * (frame_size_y//10) #to try and stop spiraling

steps_since_food1 = 0
steps_since_food2 = 0

# Q-Learning parameters
alpha = 0.1 #learning rate
gamma = 0.9 #discount factor
epsilon = 1.0 #exploration vs exploitation
epsilon2 = 1.0
epsilon_min = 0.001
epsilon_decay = 0.9995

#astar bias
astar_bias = 0.7
astar_bias2 = 0.7
astar_bias_min = 0.005
astar_bias_decay = 0.9995

# Q-table
q_table = defaultdict(lambda: [0.0, 0.0, 0.0]) #snake 1 q table
q2_table = defaultdict(lambda: [0.0, 0.0, 0.0]) #snake 2 q table

# long term tracking analysis
total_reward = 0
episode_number = 0

#snake 2 values
total_reward2 = 0
episode_number2 = 0

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()


# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]
s1_obstacles = []

food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
food_spawn = True

direction = 'RIGHT'
change_to = direction

#snake 2 variables
snake2_pos = [200, 50]
snake2_body = [[200, 50], [200+10, 50], [200+(2*10), 50]]
direction2 = 'LEFT'

score = 0
score2 = 0
game_length = 0

data_score = []
data_episode = []
data_reward = []
data_length = []

data_score2 = []
data_episode2 = []
data_reward2 = []
data_length2 = []

def start_menu():
    global mode_selected, game_mode
    font = pygame.font.SysFont('times new roman', 20)
    line1 = font.render('Press 1 to train 1 snake', True, white)
    line2 = font.render('Press 2 to train 2 snake', True, white)
    line3 = font.render('Press 3 to train 1 snake with obstacles', True, white)

    line1_x = (frame_size_x - line1.get_width()) // 2
    line1_y = (frame_size_y // 2) - 60
    line2_x = (frame_size_x - line2.get_width()) // 2
    line2_y = (frame_size_y // 2) - 20
    line3_x = (frame_size_x - line3.get_width()) // 2
    line3_y = (frame_size_y // 2) + 20

    while not mode_selected:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN: #when the user presses a key and releases it
                if event.key == pygame.K_1: #checks if the key was 1 or 2 and triggers the correct game mode
                    game_mode = 1
                    mode_selected = True
                elif event.key == pygame.K_2:
                    game_mode = 2
                    mode_selected = True
                elif event.key == pygame.K_3:
                    game_mode = 3
                    mode_selected = True
        #displays the text we wrote earlier
        game_window.fill(black)
        game_window.blit(line1, (line1_x, line1_y))
        game_window.blit(line2, (line2_x, line2_y))
        game_window.blit(line3, (line3_x, line3_y))
        pygame.display.flip()

# Game Over
def game_over():
    global snake_pos, snake_body, food_pos, food_spawn, direction, score, epsilon, total_reward, episode_number, \
        game_length, game_mode, snake2_pos, snake2_body, direction2, epsilon2, episode_number2, score2, total_reward2, \
        astar_bias, astar_bias2, steps_since_food1, steps_since_food2, s1_obstacles
    my_font = pygame.font.SysFont('times new roman', int(frame_size_x * 0.125))
    game_over_surface = my_font.render('YOU DIED', True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x/2, frame_size_y/4)
    game_window.fill(black)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(0, red, 'times', 20)
    pygame.display.flip()
    #time.sleep(.01) #3->1
    # change the game over so it just automatically restarts after game over instead of quitting
    snake_pos = [100, 50]
    snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]


    if game_mode == 1:
        print(f"Agent died with a total reward of {total_reward:.1f} on episode {episode_number}")
        # adjusts epsilon for the next round
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        astar_bias = max(astar_bias_min, astar_bias * astar_bias_decay)

        data_score.append(score)
        data_reward.append(total_reward)
        data_episode.append(episode_number)
        data_length.append(game_length)

        score = 0
        steps_since_food1 = 0
        game_length = 0
        # resets reward and increment episode
        total_reward = 0
        episode_number += 1
    elif game_mode == 2:
        print(f"Snake 2 Agent died with a total reward of {total_reward2:.1f} on episode {episode_number2}")
        snake2_pos = [200, 50]
        snake2_body = [[200, 50], [200 + 10, 50], [200 + (2 * 10), 50]]
        direction2 = 'LEFT'
        # adjusts epsilon for the next round
        epsilon2 = max(epsilon_min, epsilon2 * epsilon_decay)
        astar_bias2 = max(astar_bias_min, astar_bias2 * astar_bias_decay)

        data_score2.append(score2)
        data_reward2.append(total_reward2)
        data_episode2.append(episode_number2)
        data_length2.append(game_length)

        score2 = 0
        game_length = 0
        steps_since_food1 = 0
        steps_since_food2 = 0
        # resets reward and increment episode
        total_reward2 = 0
        episode_number2 += 1
    elif game_mode == 3:
        print(f"Snake Agent died with a total reward of {total_reward:.1f} on episode {episode_number}")
        #spawns the obstacles in a new spot every death
        s1_obstacles = spawn_obstacles(frame_size_x, frame_size_y, snake_body)
        # adjusts epsilon for the next round
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        astar_bias = max(astar_bias_min, astar_bias * astar_bias_decay)

        data_score.append(score)
        data_reward.append(total_reward)
        data_episode.append(episode_number)
        data_length.append(game_length)

        score = 0
        steps_since_food1 = 0
        game_length = 0
        # resets reward and increment episode
        total_reward = 0
        episode_number += 1
    food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
    while food_pos in snake_body or food_pos in s1_obstacles or food_pos in snake2_body:
        food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
    food_spawn = True
    direction = 'RIGHT'



# Score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    display_score = score if game_mode in (1,3) else score2
    score_surface = score_font.render('Score : ' + str(display_score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (int(frame_size_x*0.15), 15)
    else:
        score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)
    game_window.blit(score_surface, score_rect)
    # pygame.display.flip()

#A* function that the agent is going to use
def astar(snake_body, food_pos, frame_size_x, frame_size_y, extra_obstacles=None):
    if extra_obstacles is None:
        extra_obstacles = []
    head = tuple(snake_body[0])
    goal = tuple(food_pos)
    body_set = set(tuple(b) for b in snake_body[1:]) #set of locations of the snake's body (excluding the head)
    body_set.update(tuple(o) for o in extra_obstacles) #adds the other snakes body into the set of locations

    def heuristic(a, b): #caluclates the heuristics of any given position
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) #manhatten distance (horizontal + vertical)

    open_set = []
    heapq.heappush(open_set, (0, head))
    came_from = {}
    g_score = {head: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal: #reconstuct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path #first element is the next position to move to

        for dx, dy in [(0, -10), (0, 10), (-10, 0), (10, 0)]: #the numbers represent the 4 directions the snake could take
            neighbor = current[0] + dx, current[1] + dy
            #checks if neighbor is inside the bounds of the window
            if not (0 <= neighbor[0] < frame_size_x and 0 <= neighbor[1] < frame_size_y):
                continue
            #checks if neighbor is in the body collision
            if neighbor in body_set:
                continue

            tentative_g = g_score[current] + 1 #increment step count
            #continue if we haven't seen this position before or if we can get to it quicker than before
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal) #f score is calculated by steps to get here and estimated steps left
                heapq.heappush(open_set, (f, neighbor))
    return [] #no path found

#method to get the current state of the snake's head
def get_state(snake_pos, snake_body, direction, food_pos, extra_obstacles=None):
    #uses the direction the snake is going in to identify the cells straight, left, and right
    if extra_obstacles is None:
        extra_obstacles = []

    head_x = snake_pos[0]
    head_y = snake_pos[1]
    if direction == "RIGHT":
        straight = (head_x + 10, head_y)
        left = (head_x, head_y - 10)
        right = (head_x, head_y + 10)
    elif direction == "LEFT":
        straight = (head_x - 10, head_y)
        left = (head_x, head_y + 10)
        right = (head_x, head_y - 10)
    elif direction == "UP":
        straight = (head_x, head_y - 10)
        left = (head_x - 10, head_y)
        right = (head_x + 10, head_y)
    elif direction == "DOWN":
        straight = (head_x, head_y + 10)
        left = (head_x + 10, head_y)
        right = (head_x - 10, head_y)
    #defaults the danger values
    all_obstacles = set(tuple(b) for b in snake_body)
    all_obstacles.update(tuple(o) for o in extra_obstacles)

    #calculates if there is danger in each direction
    danger_straight = not (0 <= straight[0] < frame_size_x and 0 <= straight[1] < frame_size_y) or straight in all_obstacles
    danger_left = not (0 <= left[0] < frame_size_x and 0 <= left[1] < frame_size_y) or left in all_obstacles
    danger_right = not (0 <= right[0] < frame_size_x and 0 <= right[1] < frame_size_y) or right in all_obstacles

    #calculates the direction the apple is in relative to the snake head
    diff_x = food_pos[0] - snake_pos[0] #(+) -> right, (-) -> left
    diff_y = food_pos[1] - snake_pos[1] #(+) -> down, (-) -> up
    #checks whether the apple is farther vertically or horizontally
    if abs(diff_x) >= abs(diff_y):
        if diff_x > 0:
            food_direction = "RIGHT"
        else:
            food_direction = "LEFT"
    else:
        if diff_y > 0:
            food_direction = 'DOWN'
        else:
            food_direction = 'UP'

    return (danger_straight, danger_left, danger_right, direction, food_direction)

# method to update the q-table's values based off the result of the action taken
def update_q_table(state, action, reward, next_state, table):
    #gets the variables for bellman equation
    action_index = ['straight', 'left', 'right'].index(action)
    current_q = table[state][action_index]
    max_future_q = max(table[next_state]) #gets the best future reward q-value

    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q) #bellman equation
    table[state][action_index] = new_q #writes the new q value to the table

#helper method for both snakes to use instead of copy and pasting
def choose_action(state, table, eps, astar_rec, astar_bias):
    # Q-learning action selection
    roll = random.random()  # number to compare against epsilon
    if roll < eps:
        # exploring: pick a random action to learn the q-values for this action in this state
        action = random.choice(['straight', 'left', 'right'])
    else:
        bias_roll = random.random()
        #picks astar's reccomendation early on
        if bias_roll < astar_bias:
            action = astar_rec
        else:
            # exploiting: picking the best action based off the Q-table
            q_values = table[state]
            best_q = max(q_values)
            # if multiple actions share the same q_value then just do what A* recommends
            if q_values.count(best_q) > 1:
                action = astar_rec
            else:
                action_index = q_values.index(best_q)
                action = ['straight', 'left', 'right'][action_index]

    return  action

#helper method for astar instead of copy and pasting code
def get_astar_rec(path, pos, current_direction):
    if not path:
        return 'straight'

    next_pos = path[0]
    dx = next_pos[0] - pos[0]
    dy = next_pos[1] - pos[1]

    # converts the astar direction into a relative direction for the agent to use for q-learning
    if current_direction == 'RIGHT':
        if dx == 10: return 'straight'
        elif dy == 10: return 'right'
        elif dy == -10: return 'left'
    elif current_direction == 'LEFT':
        if dx == -10: return 'straight'
        elif dy == 10: return 'left'
        elif dy == -10: return 'right'
    elif current_direction == 'UP':
        if dx == 10: return 'right'
        elif dx == -10: return 'left'
        elif dy == -10: return 'straight'
    elif current_direction == 'DOWN':
        if dx == 10: return 'left'
        elif dx == -10: return 'right'
        elif dy == 10: return 'straight'
    return 'straight'

#converts the relative action into the absolute direction
def apply_action(action, current_direction):
    if action == 'straight':
        return current_direction
    turn_left = {'RIGHT': 'UP', 'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT'}
    turn_right = {'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP', 'UP': 'RIGHT'}
    if action == 'left':
        return turn_left[current_direction]
    if action == 'right':
        return turn_right[current_direction]

#helper method that returns the new position of the snake's movement
def move_snake(pos, direction):
    pos = list(pos)
    if direction == 'UP': pos[1] -= 10
    if direction == 'DOWN': pos[1] += 10
    if direction == 'RIGHT': pos[0] += 10
    if direction == 'LEFT': pos[0] -= 10
    return pos

# method to save the q table data to the computer
def save_data(filename="snake_save.json"):
    #global variables that are going to be saved
    global q_table, epsilon, episode_number, data_score, data_episode, data_reward, data_length, astar_bias
    if game_mode == 1:
        q_table_serializable = {str(k): v for k, v in q_table.items()} # converts the q table into a string
        #bundles all the data into one dictionary
        save_bundle = {
            "q_table": q_table_serializable,
            "epsilon": epsilon,
            "astar_bias": astar_bias,
            "episode_number": episode_number,
            "data_score": data_score,
            "data_episode": data_episode,
            "data_reward": data_reward,
            "data_length": data_length,
        }
        #writes the data to the file
        with open(filename, 'w') as f:
            json.dump(save_bundle, f)
        print(f"[+] Snake 1 data saved to {filename}")

    elif game_mode == 2:
        global q2_table, epsilon2, episode_number2, data_score2, data_episode2, data_reward2, data_length2, astar_bias2
        q2_table_serializable = {str(k): v for k, v in q2_table.items()}
        save_bundle2 = {
            "q_table": q2_table_serializable,
            "epsilon": epsilon2,
            "astar_bias": astar_bias2,
            "episode_number": episode_number2,
            "data_score": data_score2,
            "data_episode": data_episode2,
            "data_reward": data_reward2,
            "data_length": data_length2,
        }
        with open('snake2_save.json', 'w') as f:
            json.dump(save_bundle2, f)
        print(f"[+] Snake 2 data saved to snake2_save.json")

    elif game_mode == 3:
        q3_table_serializable = {str(k): v for k, v in q_table.items()}  # converts the q table into a string
        # bundles all the data into one dictionary
        save_bundle = {
            "q_table": q3_table_serializable,
            "epsilon": epsilon,
            "astar_bias": astar_bias,
            "episode_number": episode_number,
            "data_score": data_score,
            "data_episode": data_episode,
            "data_reward": data_reward,
            "data_length": data_length,
        }
        # writes the data to the file
        with open('snake3_save.json', 'w') as f:
            json.dump(save_bundle, f)
        print(f"[+] Snake 1 data saved to snake3_save.json")

# method to load the q table from a local computer file
def load_data(filename="snake_save.json"):
    #load snake 1 data
    # global variables that are going to be loaded
    global q_table, epsilon, episode_number, data_score, data_episode, data_reward, data_length, astar_bias
    if game_mode == 1 or game_mode == 2:
        #if a file doesn't exist
        if not os.path.exists(filename):
            print("[!] Save data does not exist")
            return
        with open(filename, 'r') as f: # reads the file and converts it back into a variable
            save_bundle = json.load(f)
        raw_q = save_bundle["q_table"] #pulls out the q table
        for k, v in raw_q.items(): # loops through and saves the values for each state
            parsed_key = ast.literal_eval(k)
            q_table[parsed_key] = v
        epsilon = save_bundle["epsilon"]
        astar_bias = save_bundle["astar_bias"]
        episode_number = save_bundle["episode_number"]
        data_score = save_bundle["data_score"]
        data_episode = save_bundle["data_episode"]
        data_reward = save_bundle["data_reward"]
        data_length = save_bundle["data_length"]
        print('[+] Snake 1 data loaded')

    #load snake 2 data
    elif game_mode == 2:
        global q2_table, epsilon2, episode_number2, data_score2, data_episode2, data_reward2, data_length2, astar_bias2
        if not os.path.exists('snake2_save.json'):
            print("[!] Snake 2 save data does not exist")
        else:
            with open('snake2_save.json', 'r') as f:
                save_bundle2 = json.load(f)
            for k, v in save_bundle2['q_table'].items():
                q2_table[ast.literal_eval(k)] = v
            epsilon2 = save_bundle2["epsilon"]
            astar_bias2 = save_bundle2["astar_bias"]
            episode_number2 = save_bundle2["episode_number"]
            data_score2 = save_bundle2["data_score"]
            data_episode2 = save_bundle2["data_episode"]
            data_reward2 = save_bundle2["data_reward"]
            data_length2 = save_bundle2["data_length"]
            print("[+] Snake 2 data loaded")

    #load snake 3 data
    elif game_mode == 3:
        # if a file doesn't exist
        if not os.path.exists('snake3_save.json'):
            print("[!] Save data does not exist")
            return
        with open('snake3_save.json', 'r') as f:  # reads the file and converts it back into a variable
            save_bundle = json.load(f)
        raw_q = save_bundle["q_table"]  # pulls out the q table
        for k, v in raw_q.items():  # loops through and saves the values for each state
            parsed_key = ast.literal_eval(k)
            q_table[parsed_key] = v
        epsilon = save_bundle["epsilon"]
        astar_bias = save_bundle["astar_bias"]
        episode_number = save_bundle["episode_number"]
        data_score = save_bundle["data_score"]
        data_episode = save_bundle["data_episode"]
        data_reward = save_bundle["data_reward"]
        data_length = save_bundle["data_length"]
        print('[+] Snake 1 data loaded')

#helper method to plot the data
def plot_data(episodes, scores, rewards, lengths, title_prefix):
    # Reusable chart function that works for either snake's data arrays
    def regression(x_data, y_data):
        # Computes a simple linear regression line for a scatter plot
        n = len(x_data)
        xbar = sum(x_data) / n
        ybar = sum(y_data) / n
        numer = sum(xi * yi for xi, yi in zip(x_data, y_data)) - n * xbar * ybar
        denum = sum(xi ** 2 for xi in x_data) - n * xbar ** 2
        if denum > 0:
            b = numer / denum
            a = ybar - b * xbar
            return [a + b * xi for xi in x_data]
        return None

    # Score over time
    plt.scatter(episodes, scores, label='Score Over Time', color='green', marker='o')
    fit = regression(episodes, scores)
    if fit: plt.plot(episodes, fit)
    plt.xlabel('Episode');
    plt.ylabel('Score');
    plt.title(f'{title_prefix} Score Over Time')
    plt.legend();
    plt.show()

    # Reward over time
    plt.scatter(episodes, rewards, label='Reward Over Time', color='red', marker='o')
    fit = regression(episodes, rewards)
    if fit: plt.plot(episodes, fit)
    plt.xlabel('Episode');
    plt.ylabel('Reward');
    plt.title(f'{title_prefix} Reward Over Time')
    plt.legend();
    plt.show()

    # Reward over score
    plt.scatter(scores, rewards, label='Reward Over Score', color='blue', marker='o')
    fit = regression(scores, rewards)
    if fit: plt.plot(scores, fit)
    plt.xlabel('Score');
    plt.ylabel('Reward');
    plt.title(f'{title_prefix} Reward Over Score')
    plt.legend();
    plt.show()

    # Reward over episode length
    plt.scatter(lengths, rewards, label='Reward Over Episode Length', color='yellow', marker='o')
    fit = regression(lengths, rewards)
    if fit: plt.plot(lengths, fit)
    plt.xlabel('Episode Length');
    plt.ylabel('Reward')
    plt.title(f'{title_prefix} Reward Over Episode Length')
    plt.legend();
    plt.show()

    # Time to eat (only episodes where score > 0 are meaningful here)
    clean_ep = [episodes[i] for i in range(len(episodes)) if scores[i] > 0]
    avg_times = [lengths[i] / scores[i] for i in range(len(scores)) if scores[i] > 0]
    if clean_ep:
        plt.scatter(clean_ep, avg_times, label='Time to Eat Over Episode', color='green', marker='o')
        fit = regression(clean_ep, avg_times)
        if fit: plt.plot(clean_ep, fit)
        plt.xlabel('Episode');
        plt.ylabel('Time to Eat')
        plt.title(f'{title_prefix} Scoring Speed Over Episode')
        plt.legend();
        plt.show()

#function that spawns the obstacles for the snake in 10% of the space randomely
def spawn_obstacles(frame_x, frame_y, snake_body):
    x = frame_x//10
    y = frame_y//10
    number_of_obstacles = math.floor((x*y) * 0.1)
    obstacles = []
    for i in range(number_of_obstacles):
        obst_pos = [random.randrange(0, x+1) * 10, random.randrange(0, y+1) * 10]
        while obst_pos in obstacles or obst_pos in snake_body:
            obst_pos = [random.randrange(0, x+1) * 10, random.randrange(0, y+1) * 10]
        obstacles.append(obst_pos)
    return obstacles


# Main logic
start_menu()
load_data()

if game_mode == 3: #spawn the obstacles in game mode 3
    s1_obstacles = spawn_obstacles(frame_size_x, frame_size_y, snake_body)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            save_data()
            #shows the charts for the relevant game mode
            if game_mode == 1 and data_episode:
                plot_data(data_episode, data_score, data_reward, data_length, 'Snake 1')
            elif game_mode == 2 and data_episode2:
                plot_data(data_episode2, data_score2, data_reward2, data_length2, 'Snake 2')
            elif game_mode == 3 and data_score:
                plot_data(data_episode, data_score, data_reward, data_length, 'Snake 3')
            sys.exit()


    if game_mode == 2: #snake 2 body is an obstacle in game mode 2
        s1_obstacles = snake2_body
    elif game_mode == 1:
        s1_obstacles = []

    # Snake 1 logic
    state1 = get_state(snake_pos, snake_body, direction, food_pos, s1_obstacles)

    #runs A* on snake 1 to the food
    path1 = astar(snake_body, food_pos, frame_size_x, frame_size_y, s1_obstacles)
    astar_rec1 = get_astar_rec(path1, snake_pos, direction)

    #pick snake 1's action
    action1 = choose_action(state1, q_table, epsilon, astar_rec1, astar_bias)
    #gets the absolute direction from the relative direction
    direction = apply_action(action1, direction)
    #calculate the distance of the snake from the food before the snake moves
    dist_before1 = abs(snake_pos[0] - food_pos[0]) + abs(snake_pos[1] - food_pos[1])
    #move snake 1
    snake_pos = move_snake(snake_pos, direction)
    #calculate the distance of the snake to the food after the snake moves
    dist_after1 = abs(snake_pos[0] - food_pos[0]) + abs(snake_pos[1] - food_pos[1])

    #snake growing/eating food
    snake_body.insert(0, list(snake_pos))
    s1_ate = (snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1])
    if s1_ate:
        score += 1
        food_spawn = False
        steps_since_food1 = 0
    else:
        snake_body.pop()
        steps_since_food1 += 1

    #calculate the reward for snake 1
    # gets boolean values for each outcome of snake movement
    s1_hit_wall = (snake_pos[0] < 0 or snake_pos[0] > frame_size_x - 10
                or snake_pos[1] < 0 or snake_pos[1] > frame_size_y - 10)
    s1_hit_self = any(snake_pos[0] == b[0] and snake_pos[1] == b[1] for b in snake_body[1:])
    s1_hit_s2 = game_mode == 2 and any(snake_pos[0] == b[0] and snake_pos[1] == b[1] for b in snake2_body)
    s1_spiral = steps_since_food1 >= max_steps_without_food
    s1_hit_obst = game_mode == 3 and any(snake_pos == block for block in s1_obstacles)
    # assigning reward based off of boolean values
    if s1_spiral:
        steps_since_food1 = 0
        reward1 = spiral_penalty
    elif s1_hit_wall or s1_hit_self or s1_hit_s2 or s1_hit_obst:
        reward1 = death
    elif s1_ate:
        reward1 = got_food
    else:
        distance_reward1 = food_distance if dist_after1 < dist_before1 else -food_distance
        reward1 = move_no_food + distance_reward1

    if game_mode == 1 or game_mode == 3:
        #get the next state using the bellman equation
        next_state1 = get_state(snake_pos, snake_body, direction, food_pos, s1_obstacles)
        #update snake 1's q table
        update_q_table(state1, action1, reward1, next_state1, q_table)
        total_reward += reward1

    #snake 1 death check
    s1_dead = s1_hit_wall or s1_hit_self or s1_hit_s2 or s1_spiral or s1_hit_obst
    if (game_mode == 1 or game_mode == 3) and s1_dead:
        game_over()
    elif game_mode == 2 and s1_dead:
        #in 2 snake game mode, snake 1 gets respawned
        #Defines several candidate spawn positions spread around the board as options for snake 1 so doesn't spawn on top of snake 2
        candidates = [
            ([100, 50], [[100, 50], [90, 50], [80, 50]], 'RIGHT'),
            ([260, 50], [[260, 50], [270, 50], [280, 50]], 'LEFT'),
            ([100, 190], [[100, 190], [90, 190], [80, 190]], 'RIGHT'),
            ([260, 190], [[260, 190], [270, 190], [280, 190]], 'LEFT'),
            ([100, 120], [[100, 120], [90, 120], [80, 120]], 'RIGHT'),
            ([260, 120], [[260, 120], [270, 120], [280, 120]], 'LEFT'),
        ]

        # Builds a set of all cells that are dangerous to spawn near
        s2_head = snake2_body[0]
        danger_zone = set(tuple(b) for b in snake2_body)
        for dx, dy in [(0, -10), (0, 10), (-10, 0), (10, 0)]:
            # Adds each of the 4 cells neighboring snake 2's head to the danger zone
            danger_zone.add((s2_head[0] + dx, s2_head[1] + dy))

        # Tries each candidate spawn until it finds one fully outside the danger zone
        chosen_pos, chosen_body, chosen_dir = candidates[0]  # fallback if nothing is safe
        for pos, body, d in candidates:
            # Checks that every cell of the candidate body is outside the danger zone
            if not any(tuple(cell) in danger_zone for cell in body):
                chosen_pos, chosen_body, chosen_dir = pos, body, d
                break
        # Applies the chosen safe spawn
        snake_pos = chosen_pos
        snake_body = chosen_body
        direction = chosen_dir
        print(f"Snake 1 respawned at {snake_pos}")

    #code to control the second snake
    if game_mode == 2:
        #gets snake 2's state, and passes snake 1's body as the extra obstacle
        state2 = get_state(snake2_pos, snake2_body, direction2, food_pos, snake_body)

        #run A* on snake 2
        path2 = astar(snake2_body, food_pos, frame_size_x, frame_size_y, snake_body)
        astar_rec2 = get_astar_rec(path2, snake2_pos, direction2)

        #pick snake 2's action
        action2 = choose_action(state2, q2_table, epsilon2, astar_rec2, astar_bias2)
        #converts the relative action to the absolute direction
        direction2 = apply_action(action2, direction2)

        # calculate the distance of the snake from the food before the snake moves
        dist_before2 = abs(snake2_pos[0] - food_pos[0]) + abs(snake2_pos[1] - food_pos[1])
        # move snake 2
        snake2_pos = move_snake(snake2_pos, direction2)
        # calculate the distance of the snake to the food after the snake moves
        dist_after2 = abs(snake2_pos[0] - food_pos[0]) + abs(snake2_pos[1] - food_pos[1])

        #grow snake 2 body and check for food
        snake2_body.insert(0, list(snake2_pos))
        s2_ate = (snake2_pos[0] == food_pos[0] and snake2_pos[1] == food_pos[1])
        if s2_ate:
            score2 += 1
            food_spawn = False
            steps_since_food2 = 0
        else:
            snake2_body.pop()
            steps_since_food2 += 1

        #calculate the reward for snake 2
        s2_hit_wall = snake2_pos[0] < 0 or snake2_pos[0] > frame_size_x - 10 \
                   or snake2_pos[1] < 0 or snake2_pos[1] > frame_size_y - 10
        s2_hit_self = any(snake2_pos[0] == b[0] and snake2_pos[1] == b[1] for b in snake2_body[1:])
        s2_hit_s1 = any(snake2_pos[0] == b[0] and snake2_pos[1] == b[1] for b in snake_body)
        s2_spiral = steps_since_food2 >= max_steps_without_food
        if s2_spiral:
            steps_since_food2 = 0
            reward2 = spiral_penalty
        elif s2_hit_wall or s2_hit_self or s2_hit_s1:
            reward2 = death
        elif s2_ate:
            reward2 = got_food
        else:
            distance_reward2 = food_distance if dist_after2 < dist_before2 else -food_distance
            reward2 = move_no_food + distance_reward2

        #get snake 2's next state for the bellman equeation
        next_state2 = get_state(snake2_pos, snake2_body, direction2, food_pos, snake_body)
        #update q table
        update_q_table(state2, action2, reward2, next_state2, q2_table)
        total_reward2 += reward2

        #snake 2 dying will trigger the end of the episode and reset
        if s2_hit_wall or s2_hit_self or s2_hit_s1 or s2_spiral:
            game_over()

    # Spawning food on the screen
    if not food_spawn:
        food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
        while food_pos in snake_body or  food_pos in s1_obstacles:
            food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
    food_spawn = True

    game_length += 1

    #update the window for the next frame
    # GFX
    game_window.fill(black)
    for pos in snake_body: #draws snake 1
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

    if game_mode == 2:
        for pos in snake2_body: #draws snake 2
            pygame.draw.rect(game_window, blue, pygame.Rect(pos[0], pos[1], 10, 10))

    if game_mode == 3:
        for pos in s1_obstacles: #draws the obstacles
            pygame.draw.rect(game_window, white, pygame.Rect(pos[0], pos[1], 10, 10))

    # draws the apple
    pygame.draw.rect(game_window, red, pygame.Rect(food_pos[0], food_pos[1], 10, 10))



    show_score(1, white, 'consolas', 20)
    # Refresh game screen
    pygame.display.update()
    # Refresh rate
    fps_controller.tick(difficulty)