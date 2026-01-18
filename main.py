import torch
import random
import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from model import QNetwork  # Aapki banayi hui model file

app = FastAPI()

# --- 1. SETTINGS & MODEL LOADING ---
STATE_SIZE = 1
ACTION_SIZE = 3
model = QNetwork(STATE_SIZE, ACTION_SIZE)

# Model load (Agar training complete ho gayi hai toh)
try:
    model.load_state_dict(torch.load("nexus_model.pth"))
    model.eval()
    print("AI Model loaded successfully!")
except:
    print("Warning: nexus_model.pth nahi mili. Pehle train.py run karein.")

# Content Mapping
ACTION_MAP = {
    0: "Basic Lesson: simpler concepts to build foundation",
    1: "Intermediate Exercise: medium difficulty content",
    2: "Advanced Challenge: you're ready for tough stuff!"
}

# --- 2. DATA SCHEMAS ---
class UserEvent(BaseModel):
    user_id: int
    event_type: str  # e.g., "quiz_failed", "video_watched"
    quiz_score: float = 0.0

# --- 3. THE SMART ENDPOINTS ---

# A. EVENT LOGGING & A/B TESTING
@app.post("/log_event")
async def log_event(event: UserEvent):
    # A/B Testing: 50% users ko RL path milega, 50% ko Static
    user_group = "RL_POWERED" if event.user_id % 2 == 0 else "STATIC_CONTROL"
    
    log_entry = {
        "user_id": event.user_id,
        "group": user_group,
        "event": event.event_type,
        "score": event.quiz_score,
        "timestamp": str(datetime.datetime.now())
    }
    
    # Yahan hum recommendation logic trigger karenge agar user fail hua
    action = None
    recommendation = "Review Basics"
    
    if user_group == "RL_POWERED" and event.event_type == "quiz_failed":
        # AI se pucho kya karna hai
        state_tensor = torch.FloatTensor([event.quiz_score]).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(model(state_tensor)).item()
    else:
        # Static logic: Agar score kam hai toh hamesha Basic (0)
        action = 0 if event.quiz_score < 0.5 else 1
    recommendation = ACTION_MAP[action]
        
    return {
        "status": "Logged",
        "group": user_group,
        "recommendation": recommendation,
        "action_taken": action
    }


# B. REWARD LOGGING (For Model Feedback)
@app.post("/log_reward")
async def log_reward(user_id: int, success: bool):
    reward = 10 if success else -5
    # Industry mein yahan data database mein jata hai model retraining ke liye
    return {
        "user_id": user_id,
        "feedback_received": "Positive" if success else "Negative",
        "reward_assigned": reward
    }

# C. STATUS CHECK (Dashboard ke liye)
@app.get("/system_status")
async def status():
    return {"status": "Nexus System is Online", "model": "PyTorch DQN"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    
        
    
    