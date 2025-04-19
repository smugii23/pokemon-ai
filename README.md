A project to create an AI using reinforcement learning to make the optimal plays for Pokemon TCG Pocket.
The plan is to:
1. Create a simulation environment so that the AI can run many games.
2. Build the reinforcement learning loop:
    - Gymnasium to develop the environment
    - Create a wrapper to translate the simulation into Gymnasium format
    - Choose an algorithm that can learn from the observations (probably PPO for now)
3. Implement the training pipeline:
    - Run the training
    - Track and visualize performance
    - Evaluate the agent

In the future, I also want to implement an LLM to explain why the AI is making these decisions, or maybe explain why making
another move is a bad decision.

Currently, I'm planning to only implement a few decks into this, because I would need to manually implement the code for each
unique ability if I were to add all pokemon. I'll just implement a few meta decks that are popular.

Because there could be variations to the different decks, I'll train the data on a core 17-19 cards, and have some slight
variation of common tech cards. 
