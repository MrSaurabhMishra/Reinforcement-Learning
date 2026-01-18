🧠 Nexus: AI-Powered Adaptive Learning System
Nexus is a Reinforcement Learning (RL) based education platform designed to dynamically adjust course content difficulty based on student performance. It utilizes a Deep Q-Network (DQN) to intelligently decide whether a student requires a basic foundation, intermediate exercises, or advanced challenges.

🚀 Key Features
Adaptive Content Delivery: Uses RL to recommend content levels (Basic, Intermediate, Advanced) based on real-time quiz scores.

A/B Testing Framework: Includes logic to compare RL-powered recommendations against a static control group by splitting users based on ID.

Real-time Dashboard: A Streamlit interface to track class-wide completion rates, student scores, and the RL agent's learning progress.

Simulation Environment: A custom UserSimulation that models how different student knowledge levels respond to varying difficulty spikes to train the AI.

Dual-Framework Support: Provides neural network implementations for both PyTorch and TensorFlow.

🛠️ Tech Stack
Deep Learning: PyTorch and TensorFlow.

API Framework: FastAPI.

Data Visualization: Streamlit, Plotly, and Pandas.

RL Algorithm: Deep Q-Learning (DQN).

📂 Project Structure
model.py / modelTe.py: Defines the Neural Network architecture and RL Agent logic in PyTorch and TensorFlow respectively.

train.py: The training loop where the agent learns by interacting with the student simulation for a set number of episodes.

simulation.py: Logic for a simulated student that gains "knowledge" from lessons but receives negative rewards if content is too difficult.

main.py: FastAPI production script for logging events and serving real-time AI recommendations.

dashboard.py: Interactive UI to visualize performance metrics and test the API endpoints live.

⚙️ Installation & Setup
1. Train the AI Agent
Before running the system, the agent must learn the optimal teaching strategy through simulation:

Bash

python train.py
This will generate nexus_model.pth after 500 training episodes.

2. Launch the Backend API
Start the FastAPI server to handle event logging and A/B testing:

Bash

python main.py
The server typically runs on http://localhost:8000.

3. Run the Dashboard
Open the visual analytics interface:

Bash

streamlit run dashboard.py
📊 How the RL Works
State: The student's current quiz score or simulated knowledge level.

Action: The AI chooses between 3 difficulty levels:

0: Basic Lesson (simpler concepts).

1: Intermediate Exercise (medium difficulty).

2: Advanced Challenge (tough stuff).

Reward: * Positive rewards (e.g., +20 or +10) for successful progression or passing hard tasks.

Negative rewards (e.g., -5 or -2) if the content is too difficult for the student's level.

📝 API Endpoints
POST /log_event: Receives user performance data and returns a tailored recommendation based on the RL model or static logic.

POST /log_reward: Updates the system with feedback on whether the recommendation was successful for future retraining.

GET /system_status: Checks if the Nexus system and the PyTorch DQN model are online.
