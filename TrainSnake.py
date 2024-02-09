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
