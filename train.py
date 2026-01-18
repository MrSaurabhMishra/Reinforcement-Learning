from model import NexusAgent
from simulation import UserSimulation # Purani simulation file
import numpy as np
import torch
import random
sim = UserSimulation()
agent = NexusAgent(state_size=1, action_size=3) # State: knowledge_level, Actions: 3

episodes = 500
epsilon = 1.0 # Starting exploration

for e in range(episodes):
    
    # Randomly start at different knowledge levels so the AI sees all scenarios
    sim.knowledge_level = random.uniform(0, 1.5) 
    state = np.array([sim.knowledge_level])
    # ... rest of your loop ...
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state, epsilon)
        next_level, reward, done = sim.respond_to_content(action)
        
        next_state = np.array([next_level])
        agent.train(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    epsilon = max(0.01, epsilon * 0.99) # Dhire dhire explore kam karo
    if e % 50 == 0:
        print(f"Episode: {e}, Total Reward: {total_reward}")

# Model save karo Week 3 ke liye
torch.save(agent.model.state_dict(), "nexus_model.pth")
print("Model Trained and Saved!")