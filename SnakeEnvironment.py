class Snake(gym.Env):

    def __init__(self, visuals=False):
        super(Snake, self).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=-500, high=500, shape=(5+20,), dtype=np.float32)
        self.show_env = visuals
        self.timestep = 0
        self.timestep_max = 2000

    def step(self, action):
        self.prev_actions.append(action)
        self.timestep += 1

#         print("Snake Direction Initial: ",self.head_direction)
#         print("Snake Head Initial: ",self.snake_head)
        
        if action == 0:
            self.head_direction = (self.head_direction + 1) % 4
        elif action == 1:
            self.head_direction = (self.head_direction - 1) % 4
        elif action == 2:
            pass
        
        if self.head_direction == 0:
            self.snake_head[0] += 10
        elif self.head_direction == 1:
            self.snake_head[1] += 10
        elif self.head_direction == 2:
            self.snake_head[0] -= 10
        elif self.head_direction == 3:
            self.snake_head[1] -= 10
            
#         print("Snake Direction After: ",self.head_direction)
#         print("Snake Head After: ",self.snake_head)

        # Increase Snake length on eating apple
        apple_reward = 0
        if self.snake_head == self.apple_position:
            self.apple_count += 1
            self.apple_position = self.apple_position_list[self.apple_count]
            self.snake_position.insert(0,list(self.snake_head))
            apple_reward = 500
            if len(self.snake_position) == MAX_SNAKE_LEN:
                self.reward += 500
                if self.show_env:
                    viz_game_win(self.apple_count)
            
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
            
        
        if self.show_env:
            viz_game(self.snake_position, self.apple_position, self.snake_head, self.head_direction)
        
        
        # On collision end episode and print the score
        if boundary_collision(self.snake_head) == 1 or self_collision(self.snake_position) == 1:
            if self.show_env:
                viz_game_over(self.apple_count)
            self.done = True
            self.reward = -100


        # REWARD DYNAMICS
        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
        self.reward =  (150 - euclidean_dist_to_apple)/10 + apple_reward

        ## if snake dies
#         if self.done:
#             self.reward = -100                                     
        info = {}
    
        if self.timestep == self.timestep_max:
            self.done = True

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, info

    def reset(self):
        
        # Initial Snake and Apple position
        self.snake_position = [[70,80],[60,80],[50,80]]
        self.apple_count = 0
        self.apple_position_list = [[110,60],[50,40],[50,80],[60,110],[70,110],[80,90],[110,60],\
                                    [40,50],[40,80],[40,110],[120,20],[120, 120],[30,120],[90,40],[30,20]]
        self.apple_position = self.apple_position_list[self.apple_count]
        
        self.prev_head_direction = 1
        self.head_direction = 0
        self.snake_head = [70,80]                                                           # head_x and head_y

        self.prev_reward = 0
        self.timestep = 0

        self.done = False                                                                     # Reset the environment after every episode and set done as False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen = 20)
        for i in range(20):
            self.prev_actions.append(-1) 

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation
