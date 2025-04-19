# train.py (Corrected Env Handling and Saving)

import os
import time
import numpy as np
import multiprocessing # Import multiprocessing
from tqdm.auto import tqdm # Import tqdm
import traceback # Import traceback for detailed error printing

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback # Import CheckpointCallback directly
from stable_baselines3.common.monitor import Monitor # Import Monitor
# Import the wrapper
from sb3_contrib.common.wrappers import ActionMasker

from rl_env import PokemonTCGPocketEnv # Assuming rl_env.py is accessible

# --- Configuration ---
TOTAL_TIMESTEPS = 10000000   # KEEP LOW FOR DEBUGGING RESET ISSUE FIRST!
NUM_ENVIRONMENTS = 1     # KEEP AS 1 FOR DEBUGGING RESET ISSUE FIRST!
LOG_INTERVAL = 10
MODEL_SAVE_FREQ = 3000000   # Lower freq for debugging saves
LOG_PATH = "logs/"
MODEL_SAVE_PATH = "models/pokemon_maskable_ppo_agent"
VEC_NORMALIZE_SAVE_PATH = "models/vecnormalize.pkl"

# --- PPO Hyperparameters ---
HYPERPARAMS = {
    "learning_rate": 1e-4,
    "n_steps": 1024,          # Lower n_steps might help isolate reset issues faster
    "batch_size": 64,
    "n_epochs": 4,           # Lower epochs might speed up debugging iterations slightly
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128])) # Smaller network for faster init
}

# --- Tqdm Callback (Remains the same) ---
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
    def _on_step(self) -> bool:
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None

# --- Environment Creation Function (Remains the same) ---
def make_env(rank, seed=21):
    """Utility function for multiprocessed env."""
    def _init():
        env = PokemonTCGPocketEnv(render_mode=None)
        env = ActionMasker(env, lambda env_instance: env_instance.action_mask_fn())
        env = Monitor(env)
        return env
    return _init

# --- Main Training Function (Modified Env Handling) ---
def train_agent():
    """Trains the agent using the standard Stable Baselines3 approach with callbacks."""
    print(f"Creating VecEnv with {NUM_ENVIRONMENTS} environments...")
    env_fns = [make_env(i) for i in range(NUM_ENVIRONMENTS)]
    if NUM_ENVIRONMENTS == 1:
        print("Using DummyVecEnv.")
        # Create the base VecEnv first
        base_vec_env = DummyVecEnv(env_fns)
    else:
        print(f"Using SubprocVecEnv.")
        start_method = 'fork' if os.name != 'nt' else 'spawn'
        try:
             base_vec_env = SubprocVecEnv(env_fns, start_method=start_method)
        except Exception as e:
             print(f"Warning: Failed with start_method='{start_method}' ({e}). Trying 'spawn'.")
             base_vec_env = SubprocVecEnv(env_fns, start_method='spawn')

    # --- Load or Initialize Agent ---
    # Define paths
    model_file_path = MODEL_SAVE_PATH + ".zip"
    vec_normalize_path = VEC_NORMALIZE_SAVE_PATH

    # This variable will hold the *final* environment passed to the agent (potentially normalized)
    # Initialize it to the base VecEnv
    env_for_agent = base_vec_env

    if os.path.exists(model_file_path) and os.path.exists(vec_normalize_path):
        print(f"Loading existing model from {model_file_path}...")
        print(f"Loading VecNormalize stats from {vec_normalize_path}...")
        # Load VecNormalize, wrapping the base_vec_env
        # Pass normalize_reward=False if you don't want reward normalization loaded/applied
        env_for_agent = VecNormalize.load(vec_normalize_path, base_vec_env)
        # Set training=True if you are continuing training
        env_for_agent.training = True
        # Do NOT reset VecNormalize here, SB3's learn() will handle it
        print("VecNormalize loaded.")

        # Load the model, passing the (now normalized) env_for_agent
        model = MaskablePPO.load(
            model_file_path,
            env=env_for_agent, # Pass the loaded VecNormalize instance
            tensorboard_log=LOG_PATH,
            verbose=1,
            # Pass hyperparams that might change if needed for continued training
            # custom_objects={'learning_rate': HYPERPARAMS['learning_rate'], ...}
        )
        print(f"Agent loaded. Continuing training for {TOTAL_TIMESTEPS} additional timesteps...")
        # Optional: Reset num_timesteps if TOTAL_TIMESTEPS means additional steps
        # model.num_timesteps = 0
    else:
        print("No existing model or VecNormalize stats found.")
        print("Wrapping environment with VecNormalize for new training...")
        # Wrap the base_vec_env with VecNormalize for new training
        env_for_agent = VecNormalize(base_vec_env, norm_obs=True, norm_reward=False, gamma=HYPERPARAMS["gamma"])
        print(f"VecNormalize wrapper applied. Gamma: {env_for_agent.gamma}")

        print("Initializing new agent...")
        model = MaskablePPO("MlpPolicy", env_for_agent, verbose=1, tensorboard_log=LOG_PATH, **HYPERPARAMS)
        print("New agent initialized.")

    # --- Callbacks ---
    # Callbacks should operate on the final environment (env_for_agent)
    tqdm_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(MODEL_SAVE_FREQ // NUM_ENVIRONMENTS, 1),
        save_path=os.path.dirname(MODEL_SAVE_PATH),
        name_prefix=os.path.basename(MODEL_SAVE_PATH),
        save_replay_buffer=False,
        save_vecnormalize=True, # Save the VecNormalize wrapper
        verbose=1
    )
    callback_list = [tqdm_callback, checkpoint_callback]

    # --- Training ---
    try:
        print(f"Starting training for {TOTAL_TIMESTEPS} total timesteps...")
        # The model already has the correct env reference (env_for_agent)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            log_interval=LOG_INTERVAL,
            reset_num_timesteps=not (os.path.exists(model_file_path) and os.path.exists(vec_normalize_path)) # Reset counter if not loading
        )
        print("\nTraining finished.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during model.learn(): {e}")
        traceback.print_exc() # Print full traceback
    finally:
        # --- Final Saving ---
        # Ensure we have a reference to the potentially normalized environment
        final_env = model.get_env()

        print(f"\nSaving final model to {MODEL_SAVE_PATH}.zip...")
        model.save(MODEL_SAVE_PATH)
        print("Model saved.")

        # Save VecNormalize statistics *only if* the environment is wrapped
        if isinstance(final_env, VecNormalize):
            print(f"Saving final VecNormalize statistics to {VEC_NORMALIZE_SAVE_PATH}...")
            final_env.save(VEC_NORMALIZE_SAVE_PATH)
            print("VecNormalize statistics saved.")
        else:
            print("Final environment is not VecNormalize, skipping stats saving.")

        # Close the base environment (VecNormalize doesn't have a close method itself)
        # Access the underlying VecEnv if needed
        if hasattr(final_env, 'close'): # Handles VecNormalize and base VecEnvs
             final_env.close()
        elif hasattr(final_env, 'venv'): # If final_env is VecNormalize, close its venv
             final_env.venv.close()
        print("Environments closed.")

# --- Main Execution Block (Remains the same) ---
if __name__ == "__main__":
    # ... (multiprocessing setup) ...
    multiprocessing.freeze_support()
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    train_agent()
    print("\n --- Training script finished ---")
    print(f"To view training logs, run: tensorboard --logdir {LOG_PATH}")
    print(f"Final model saved at: {MODEL_SAVE_PATH}.zip")
    print(f"Final VecNormalize stats saved at: {VEC_NORMALIZE_SAVE_PATH}")