A project to create an AI using reinforcement learning to make the optimal plays for Pokemon TCG Pocket.
The plan is to:
1. Create a simulation environment so that the AI can run many games.
2. Build the reinforcement learning loop:
    - Gymnasium to develop the environment
    - Create a wrapper to translate the simulation into Gymnasium format
    - Choose an algorithm that can learn from the observations (probably PPO)
3. Implement the training pipeline:
    - Run the training
    - Track and visualize performance
    - Evaluate the agent

In the future, I also want to implement an LLM to explain why the AI is making these decisions, or maybe explain why making
another move is a bad decision.
