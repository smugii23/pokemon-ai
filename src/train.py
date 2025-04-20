# train.py (Modified for Self-Play Env and Opponent Checkpoints)

import os
import sys
import time
import numpy as np
import multiprocessing # Import multiprocessing
from tqdm.auto import tqdm # Import tqdm
import traceback # Import traceback for detailed error printing
import shutil # To clean checkpoint dir if needed
import functools
from typing import Optional

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor # Import Monitor
# Import the wrapper
from sb3_contrib.common.wrappers import ActionMasker
# Import env_util for make_vec_env
from stable_baselines3.common.env_util import make_vec_env

from rl_env import PokemonTCGPocketEnv, ACTION_MAP, ACTION_PASS # Assuming rl_env.py is accessible

# --- Configuration ---
TOTAL_TIMESTEPS = 10000000   # Adjust as needed for full training
NUM_ENVIRONMENTS = 1     # Increase for parallel training (e.g., 4, 8, 16)
LOG_INTERVAL = 10
MODEL_SAVE_FREQ = 500000   # How often to save the *main* agent checkpoint
LOG_PATH = "logs/pokemon_ppo_selfplay_v2" # New log dir recommended
MODEL_SAVE_PATH = "models/pokemon_ppo_selfplay_v2/agent" # Base path for agent saves
CHECKPOINT_DIR = "models/ppo_checkpoints_v2" # Directory for historical opponents
VEC_NORMALIZE_SAVE_PATH = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "vecnormalize_final.pkl") # Path for final stats

# --- Opponent Checkpoint Configuration ---
OPPONENT_CHECKPOINT_FREQ = 500000 # How often to save a version for the opponent pool (adjust)
CLEAN_CHECKPOINTS_ON_START = False # Set to True to remove old checkpoints before new training

# --- PPO Hyperparameters (Keep or adjust) ---
HYPERPARAMS = {
    "learning_rate": 1e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
}

# --- Tqdm Callback (Remains the same) ---
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.last_print_step = 0 # Track when we last printed debug info
        print("TqdmCallback Initialized")

    def _on_training_start(self):
        print("TqdmCallback: _on_training_start CALLED")
        # Try forcing ascii=True if rendering characters are an issue
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", ascii=False, file=sys.stdout) # Keep stdout for now

    def _on_step(self) -> bool:
        # Update the progress bar's value
        if self.pbar:
            self.pbar.n = self.num_timesteps
            # --- TRY THIS: Explicitly refresh ---
            self.pbar.refresh()
            # --- End Change ---

        # Print debug info less frequently
        if self.num_timesteps - self.last_print_step >= 10000: # Print every 10k steps
            print(f"\n[Callback Debug] TqdmCallback: _on_step CALLED (step {self.num_timesteps})") # Add newline
            self.last_print_step = self.num_timesteps

        return True

    def _on_training_end(self):
        print("TqdmCallback: _on_training_end CALLED")
        if self.pbar:
            self.pbar.close()
            self.pbar = None


# --- Environment Creation Function (MODIFIED TO PASS ARGS) ---
def make_env(rank: int, seed: int = 0, env_kwargs: Optional[dict] = None):
    """Utility function for multiprocessed env. Accepts rank and seed."""
    if env_kwargs is None: env_kwargs = {}
    def _init():
        # 1. Create the base environment
        env = PokemonTCGPocketEnv(render_mode=None, **env_kwargs)

        # 2. Define the mask function - it will operate on the base env
        def get_action_mask_for_base_env(base_env_instance: PokemonTCGPocketEnv):
             agent_player = base_env_instance.agent_player
             action_space = base_env_instance.action_space # Use base env's action space
             if agent_player is None:
                 pass_id = ACTION_MAP.get(ACTION_PASS)
                 mask = np.zeros(action_space.n, dtype=bool)
                 if pass_id is not None: mask[pass_id] = True
                 return mask
             # Call the method on the base env instance passed to the function
             return base_env_instance.action_mask_fn(agent_player)

        # 3. Apply ActionMasker, passing the base env and the mask function
        env = ActionMasker(env, get_action_mask_for_base_env)

        # 4. Apply Monitor (wraps the ActionMasker-wrapped env)
        env = Monitor(env)

        # Seeding is typically handled by SB3 VecEnv setup

        return env
    return _init

# --- Main Training Function (Modified Env/Callback Handling) ---
def train_agent():
    """Trains the agent using the standard Stable Baselines3 approach with callbacks."""
    # --- Setup Directories (remains the same) ---
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # --- Clean Checkpoints (remains the same) ---
    if CLEAN_CHECKPOINTS_ON_START:
        # ... (cleaning logic) ...
        pass

    # --- Create VecEnv (CORRECTED CALL) ---
    print(f"Creating VecEnv with {NUM_ENVIRONMENTS} environments...")
    env_kwargs_dict = {'opponent_checkpoints_dir': CHECKPOINT_DIR}
    vec_env_cls = SubprocVecEnv if NUM_ENVIRONMENTS > 1 else DummyVecEnv
    start_method = 'fork' if os.name != 'nt' and NUM_ENVIRONMENTS > 1 else 'spawn'

    # Create a list of functions, where each function creates one env instance
    # using the make_env helper we defined above.
    # We use partial to pre-set the env_kwargs for make_env.
    env_fns = []
    base_seed = int(time.time()) # Use a base seed for reproducibility if needed
    for i in range(NUM_ENVIRONMENTS):
        # Create a partial function for *this specific rank*
        # It pre-fills env_kwargs. The VecEnv will call this with no args.
        # Correction: The VecEnv worker calls the function it receives with no args.
        # So the lambda/partial needs to call make_env with rank and seed.
        rank_seed = base_seed + i
        # Use lambda to capture current i and rank_seed
        env_lambda = lambda rank=i, seed=rank_seed: make_env(rank=rank, seed=seed, env_kwargs=env_kwargs_dict)()
        env_fns.append(env_lambda)


    try:
        # Directly instantiate the VecEnv class with the list of lambdas
        if vec_env_cls == SubprocVecEnv:
            base_vec_env = SubprocVecEnv(env_fns, start_method=start_method)
        else:
            base_vec_env = DummyVecEnv(env_fns)
    except Exception as e:
        print(f"Error creating VecEnv: {e}")
        traceback.print_exc()
        return


    # --- Apply VecNormalize Wrapper ---
    # Wraps the VecEnv which contains Monitor/ActionMasker-wrapped Envs
    # The commented-out ActionMasker line below is CORRECTLY commented out.
    # masked_vec_env = ActionMasker(base_vec_env, get_action_masks) # This line should NOT be here

    # --- Load or Initialize Agent ---
    model_file_path = MODEL_SAVE_PATH + ".zip"
    env_for_agent = base_vec_env # Start with the base VecEnv

    if os.path.exists(model_file_path) and os.path.exists(VEC_NORMALIZE_SAVE_PATH):
        print(f"Loading existing model from {model_file_path}...")
        print(f"Loading VecNormalize stats from {VEC_NORMALIZE_SAVE_PATH}...")
        # Wrap the base_vec_env
        env_for_agent = VecNormalize.load(VEC_NORMALIZE_SAVE_PATH, base_vec_env)
        env_for_agent.training = True
        env_for_agent.norm_reward = False
        print("VecNormalize loaded.")
        model = MaskablePPO.load(
            model_file_path, env=env_for_agent, tensorboard_log=LOG_PATH, verbose=1,
            custom_objects={'learning_rate': HYPERPARAMS['learning_rate']}
        )
        print(f"Agent loaded. Continuing training...")
        reset_num_timesteps = False
    else:
        print("No existing model or VecNormalize stats found.")
        print("Wrapping environment with NEW VecNormalize...")
        # Wrap the base_vec_env
        env_for_agent = VecNormalize(base_vec_env, norm_obs=True, norm_reward=True, gamma=HYPERPARAMS["gamma"])
        print(f"VecNormalize wrapper applied. Gamma: {env_for_agent.gamma}")
        print("Initializing new agent...")
        model = MaskablePPO("MlpPolicy", env_for_agent, verbose=1, tensorboard_log=LOG_PATH, **HYPERPARAMS)
        print("New agent initialized.")
        reset_num_timesteps = True

    # --- Callbacks (remain the same) ---
    tqdm_callback = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
    agent_checkpoint_callback = CheckpointCallback(
        save_freq=max(MODEL_SAVE_FREQ // NUM_ENVIRONMENTS, 1),
        save_path=os.path.dirname(MODEL_SAVE_PATH),
        name_prefix=os.path.basename(MODEL_SAVE_PATH),
        save_replay_buffer=False, save_vecnormalize=True, verbose=1
    )
    opponent_checkpoint_callback = CheckpointCallback(
        save_freq=max(OPPONENT_CHECKPOINT_FREQ // NUM_ENVIRONMENTS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_opponent",
        save_replay_buffer=False, save_vecnormalize=True, verbose=1
    )
    callback_list = [tqdm_callback, agent_checkpoint_callback, opponent_checkpoint_callback]

    # --- Training (remains the same) ---
    try:
        print(f"Starting training towards {TOTAL_TIMESTEPS} total timesteps...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            log_interval=LOG_INTERVAL,
            reset_num_timesteps=reset_num_timesteps
        )
        print("\nTraining finished.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during model.learn(): {e}")
        traceback.print_exc()
    finally:
        # --- Final Saving (remains the same) ---
        final_env = model.get_env()
        print(f"\nSaving final model to {MODEL_SAVE_PATH}.zip...")
        model.save(MODEL_SAVE_PATH)
        print("Model saved.")
        if isinstance(final_env, VecNormalize):
            print(f"Saving final VecNormalize statistics to {VEC_NORMALIZE_SAVE_PATH}...")
            final_env.save(VEC_NORMALIZE_SAVE_PATH)
            print("VecNormalize statistics saved.")
        else: print("Final environment is not VecNormalize, skipping stats saving.")
        if final_env is not None: final_env.close()
        print("Environments closed.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set start method for multiprocessing (required on some platforms)
    # Use 'fork' on Linux/macOS if possible, 'spawn' otherwise (Windows)
    mp_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'
    multiprocessing.set_start_method(mp_start_method, force=True)

    train_agent() # Call the main training function

    print("\n --- Training script finished ---")
    # ... (print tensorboard command, save paths) ...