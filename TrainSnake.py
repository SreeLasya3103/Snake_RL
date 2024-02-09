class Net(nn.Module):
#     torch.manual_seed(5)
#     np.random.seed(5)
    def __init__(self, state_size, action_size, learning_rate):
        super(Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.state_size, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, self.action_size)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_step(self, state, target):
        self.optimizer.zero_grad()
        pred = self.forward(state)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class DDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, tau=0.01, batch_size=64, Test = False):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon if not Test else 0.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.batch_size = batch_size
        self.model = Net(self.state_size, self.action_size, self.learning_rate)
        self.target_model = Net(self.state_size, self.action_size, self.learning_rate)
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=500000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model(torch.tensor(state, dtype=torch.float)).detach().numpy())

    def replay(self):
            minibatch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                if not done:
                    next_action = np.argmax(self.model(torch.tensor(next_state, dtype=torch.float)).detach().numpy())
                    target = (reward + self.gamma * self.target_model(torch.tensor(next_state, dtype=torch.float))[0][next_action].detach().numpy())
                else:
                    target = reward
                target_f = self.model(torch.tensor(state, dtype=torch.float))
                target_f[0][action] = target

                self.model.train_step(torch.tensor(state, dtype=torch.float), target_f)
 
            #if self.epsilon > self.epsilon_min:
#                 self.epsilon *= self.epsilon_decay

            # Soft target update
            for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)


    def load(self, name):
        self.model.load_state_dict(torch.load(name))
#         self.target_model.load_state_dict(torch.load(name))
        

    def save(self, name):
        torch.save(self.target_model.state_dict(), name)

env = Snake()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DDQNAgent(state_size, action_size)
batch_size = 64
EPISODES = 50000
reward_list_training = []
apple_list_training = []
timestep_list_training = []
eps_list = []
max_apple_count = 1
reached = False

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    episode_reward = 0
    timestep = 0
    
    while(not done):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        timestep+=1
        if done:
            if e % 1000 == 0:
                print("Episode: {}/{}, Episode Reward: {}, Epsilon: {:.2}, Episode Apple Count: {}, Timestep: {}"
                      .format(e, EPISODES, episode_reward, agent.epsilon, env.apple_count, timestep))
            reward_list_training.append(episode_reward)
            apple_list_training.append(env.apple_count)
            timestep_list_training.append(timestep)
            # Saving weights
            agent.save("billodal_syadavil_akhilshr_project_ddqn_1.h5")
            break
    if len(agent.memory) > batch_size:
        agent.replay()
    
    eps_list.append(agent.epsilon)
    if len(agent.memory) > 60000:
        if not reached: # THIS WILL RUN ONLY ONCE
            train_start_at = e
            reached = True
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = agent.epsilon*((agent.epsilon_min/1)**(1/(EPISODES-train_start_at)))

env.close()
