import numpy as np

class UserSimulation:
    def __init__(self):
        self.knowledge_level = 0 
    
    def respond_to_content(self, action):
        reward = 0
        done = False
        
        if action == 2: # Hard level
            if self.knowledge_level >= 1.2: # Requirement for hard stuff
                reward = 20
                done = True
            else:
                reward = -2 # Way too hard for a beginner
                
        elif action == 0: # Basic level
            self.knowledge_level += 0.4 # Actually increasing the level
            reward = 4
            
        elif action == 1: # mid level
            if self.knowledge_level >= 0.6:
                reward = 10
                done = True
            else:
                reward = -1 
        
        return self.knowledge_level, reward, done

# Test Simulation
sim = UserSimulation()
# Index [0] is now Knowledge Level, [1] is Reward, [2] is Done
level, reward, is_done = sim.respond_to_content(action=int(np.random.choice([0,1,2])))
print(f"Level: {level}, Reward: {reward}")