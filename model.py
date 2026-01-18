import torch
import torch.nn as nn
import torch.optim as optim
import random

# Neural Network jo "Action" decide karega
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# RL Agent ka Logic
class NexusAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.criterion = nn.MSELoss()

    def get_action(self, state, epsilon):
        # Epsilon-greedy: thoda explore karega, thoda seekha hua use karega
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        
        # Target calculation ko float mein convert kar rahe hain
        target = float(reward)
        if not done:
            with torch.no_grad(): # Speed ke liye
                target = float(reward + 0.99 * torch.max(self.model(next_state.unsqueeze(0))).item())
        
        current_q = self.model(state.unsqueeze(0))[0][action]
        
        # Yahan dtype=torch.float32 likhna zaroori hai
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        loss = self.criterion(current_q, target_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()