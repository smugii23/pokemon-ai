# rl_env.py (Corrected and Refined)

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List, Optional, Dict, Tuple, Any
import copy
import glob 
import traceback # For detailed error printing
import os # For process ID and other OS-related functionality

# Attempt to import from simulator package structure
try:
    from simulator.entities import Player, GameState, PokemonCard, Card, Attack, MAX_HAND_SIZE, MAX_BENCH_SIZE, POINTS_TO_WIN
    # Import all necessary action constants
    from simulator.game import (
        Game, ACTION_PASS, ACTION_ATTACK_PREFIX, ACTION_ATTACH_ENERGY_ACTIVE,
        ACTION_ATTACH_ENERGY_BENCH_PREFIX, ACTION_PLAY_BASIC_BENCH_PREFIX,
        ACTION_USE_ABILITY_ACTIVE, ACTION_PLAY_SUPPORTER_PREFIX, ACTION_PLAY_ITEM_PREFIX,
        ACTION_ATTACH_TOOL_ACTIVE, ACTION_ATTACH_TOOL_BENCH_PREFIX, ACTION_RETREAT_TO_BENCH_PREFIX, # Added retreat
        # Import NEW setup actions directly
        ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX,
        ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX,
        ACTION_SETUP_CONFIRM_READY
    )
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecNormalize
    from sb3_contrib import MaskablePPO # Assuming MaskablePPO
except ImportError:
    # Fallback for running script directly if simulator is in the same dir (less ideal)
    print("Warning: Could not import from simulator package. Assuming simulator modules are in the current directory.")
    # Re-add the fallback definitions here if needed, but ideally the package structure works
    from simulator.entities import Player, GameState, PokemonCard, Card, Attack, MAX_HAND_SIZE, MAX_BENCH_SIZE, POINTS_TO_WIN
    # Import all necessary action constants (fallback)
    # If falling back, define the constants manually (matching game.py's lowercase)
    ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX = "setup_choose_active_"
    ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX = "setup_choose_bench_"
    ACTION_SETUP_CONFIRM_READY = "setup_confirm_ready"
    # Need to re-import other constants as well in fallback
    from simulator.game import (
        Game, ACTION_PASS, ACTION_ATTACK_PREFIX, ACTION_ATTACH_ENERGY_ACTIVE,
        ACTION_ATTACH_ENERGY_BENCH_PREFIX, ACTION_PLAY_BASIC_BENCH_PREFIX,
        ACTION_USE_ABILITY_ACTIVE, ACTION_PLAY_SUPPORTER_PREFIX, ACTION_PLAY_ITEM_PREFIX,
        ACTION_ATTACH_TOOL_ACTIVE, ACTION_ATTACH_TOOL_BENCH_PREFIX, ACTION_RETREAT_TO_BENCH_PREFIX # Added retreat
    )
    BaseAlgorithm = None
    BasePolicy = None
    VecNormalize = None
    MaskablePPO = None


# --- Constants Definition ---

# Define known cards based on example decks (expand as needed)
# IMPORTANT: These MUST match the names used in the Game's card loading logic (e.g., cards.json)
KNOWN_CARDS = ["Darkrai ex", "Giratina ex", "Professor's Research", "Poké Ball", "Red", "Giant Cape", "Rocky Helmet", "Sabrina", "Cyrus", "Mars", "Potion", "Pokémon Center Lady", "Leaf", "Team Rocket Grunt", "Iono"]
# Add a placeholder for unknown/padding cards
UNKNOWN_CARD_NAME = "UnknownCard"
ALL_CARD_NAMES = KNOWN_CARDS + [UNKNOWN_CARD_NAME]
CARD_TO_ID = {name: i for i, name in enumerate(ALL_CARD_NAMES)}
ID_TO_CARD = {i: name for i, name in enumerate(ALL_CARD_NAMES)}
NUM_KNOWN_CARDS = len(ALL_CARD_NAMES) # Total number of distinct card IDs

# Define energy types used in the game (expand as needed)
ENERGY_TYPES = ["Darkness", "Colorless"] # Add others if needed
# Add a placeholder for None/Unknown energy
NO_ENERGY_NAME = "None"
ALL_ENERGY_TYPES = ENERGY_TYPES + [NO_ENERGY_NAME]
ENERGY_TO_ID = {name: i for i, name in enumerate(ALL_ENERGY_TYPES)}
ID_TO_ENERGY = {i: name for i, name in enumerate(ALL_ENERGY_TYPES)}
NUM_ENERGY_TYPES = len(ALL_ENERGY_TYPES) # Total number of distinct energy IDs

MAX_HAND_SIZE = 10
MAX_BENCH_SIZE = 3
POINTS_TO_WIN = 3

# Environment-specific constants
MAX_ATTACKS_PER_POKEMON = 2 # Assumption: Pokemon have at most 2 attacks relevant to the state
MAX_DECK_SIZE = 20 # Use the constant imported from the game simulator
UNKNOWN_CARD_ID = CARD_TO_ID[UNKNOWN_CARD_NAME]
NO_ENERGY_ID = ENERGY_TO_ID[NO_ENERGY_NAME]

# --- Action Mapping ---
# This mapping defines the discrete action space for the RL agent.
# It translates the game's string-based actions into integer IDs.
# We build it sequentially to make index calculation easier.
_ACTION_MAP_BUILDER = {}
_current_action_id = 0

# --- Setup Actions (Refactored for Simultaneous Setup) ---
# Choose Active from Hand
for i in range(MAX_HAND_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1
# Choose Bench from Hand (can choose multiple, one action per card)
for i in range(MAX_HAND_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1
# Confirm Setup Ready
_ACTION_MAP_BUILDER[ACTION_SETUP_CONFIRM_READY] = _current_action_id
_current_action_id += 1


# --- Normal Turn Actions ---
# PASS
_ACTION_MAP_BUILDER[ACTION_PASS] = _current_action_id
_current_action_id += 1

# ATTACH ENERGY
_ACTION_MAP_BUILDER[ACTION_ATTACH_ENERGY_ACTIVE] = _current_action_id
_current_action_id += 1
for i in range(MAX_BENCH_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_ATTACH_ENERGY_BENCH_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1

# PLAY BASIC
for i in range(MAX_HAND_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_PLAY_BASIC_BENCH_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1

# ATTACK
for i in range(MAX_ATTACKS_PER_POKEMON):
    _ACTION_MAP_BUILDER[f"{ACTION_ATTACK_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1

# USE ABILITY
_ACTION_MAP_BUILDER[ACTION_USE_ABILITY_ACTIVE] = _current_action_id
_current_action_id += 1
# Add actions for using abilities from the bench
ACTION_USE_ABILITY_BENCH_PREFIX = "USE_ABILITY_BENCH_" # Ensure prefix is defined/imported
for i in range(MAX_BENCH_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_USE_ABILITY_BENCH_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1


# PLAY TRAINER (Supporter, Item, Tool) - Need to add these prefixes if not already imported
# Assuming they are imported or defined elsewhere
# PLAY SUPPORTER
for i in range(MAX_HAND_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_PLAY_SUPPORTER_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1
# PLAY ITEM
for i in range(MAX_HAND_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_PLAY_ITEM_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1
# ATTACH TOOL ACTIVE
for i in range(MAX_HAND_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_ATTACH_TOOL_ACTIVE}{i}"] = _current_action_id
    _current_action_id += 1
# ATTACH TOOL BENCH
for bench_idx in range(MAX_BENCH_SIZE):
    for hand_idx in range(MAX_HAND_SIZE):
        _ACTION_MAP_BUILDER[f"{ACTION_ATTACH_TOOL_BENCH_PREFIX}{bench_idx}_{hand_idx}"] = _current_action_id
        _current_action_id += 1

# RETREAT
for i in range(MAX_BENCH_SIZE):
    _ACTION_MAP_BUILDER[f"{ACTION_RETREAT_TO_BENCH_PREFIX}{i}"] = _current_action_id
    _current_action_id += 1


# Finalize action maps
ACTION_MAP = _ACTION_MAP_BUILDER
ID_TO_ACTION = {v: k for k, v in ACTION_MAP.items()}
# Calculate the total number of possible discrete actions based on the map
NUM_POSSIBLE_ACTIONS = _current_action_id # The next ID would be the size


ARCHETYPE_CORE = ["Darkrai ex"] * 2 + ["Giratina ex"] * 2 + ["Professor's Research"] * 2 + ["Poké Ball"] * 2 + ["Red"] + ["Giant Cape"] + ["Rocky Helmet"] + ["Sabrina"] + ["Cyrus"] + ["Mars"] + ["Potion"] + ["Pokémon Center Lady"] + ["Leaf"]
TECH_POOL = {
    "Potion": 1,
    "Pokémon Center Lady": 1,
    "Team Rocket Grunt": 1,
    "Iono": 1,
    "Giant Cape": 1,
    "Rocky Helmet": 1,
    "Red": 1,
    "Mars": 1
}
CORE_SIZE = len(ARCHETYPE_CORE)
NUM_TECH_SLOTS = MAX_DECK_SIZE - CORE_SIZE

# --- Observation Space Calculation ---
# Calculate the size required for representing a single Pokemon in the flattened state vector.
def _calculate_pokemon_obs_size(include_can_attack=True):
    # ... (this helper function remains the same) ...
    # exists (1=yes, 0=no): 1
    # card_id (index in ALL_CARD_NAMES): 1
    # current_hp: 1
    # max_hp: 1
    # energy_counts (one slot per type in ALL_ENERGY_TYPES): NUM_ENERGY_TYPES
    # is_ex (1=yes, 0=no): 1
    # type_id (index in ALL_ENERGY_TYPES): 1
    # weakness_id (index in ALL_ENERGY_TYPES): 1
    size = 1 + 1 + 1 + 1 + NUM_ENERGY_TYPES + 1 + 1 + 1
    if include_can_attack:
        # can_attack_flags (one per possible attack): MAX_ATTACKS_PER_POKEMON
        size += MAX_ATTACKS_PER_POKEMON
    return size

POKEMON_OBS_SIZE = _calculate_pokemon_obs_size(include_can_attack=True)
OPP_POKEMON_OBS_SIZE = _calculate_pokemon_obs_size(include_can_attack=False) # Opponent's can_attack status is not observed

# Calculate the total size of the flattened observation vector.
# IMPORTANT: This order MUST match the concatenation order in _get_obs()
FLATTENED_OBS_SIZE = (
    # My Hand (Card IDs, padded)
    MAX_HAND_SIZE +
    # My Game State Numbers
    1 +                         # deck_size
    1 +                         # discard_size
    1 +                         # points
    # My Energy Stand (Energy IDs)
    1 +                         # energy_available_id
    1 +                         # energy_preview_id
    # My Pokemon (Flattened)
    POKEMON_OBS_SIZE +          # active_pokemon
    MAX_BENCH_SIZE * POKEMON_OBS_SIZE + # bench_pokemon (padded)
    # Opponent Game State Numbers
    1 +                         # hand_size
    1 +                         # deck_size
    1 +                         # discard_size
    1 +                         # points
    # Opponent Energy Stand (Simplified: exists flag + preview ID)
    1 +                         # energy_available_exists (0 or 1)
    1 +                         # energy_preview_id
    # Opponent Pokemon (Flattened, less detail)
    OPP_POKEMON_OBS_SIZE +      # active_pokemon
    # --- MODIFICATION START ---
    # Replace opp_bench_size with detailed bench info
    # 1 +                         # bench_size (Simplified: only count) # REMOVED
    MAX_BENCH_SIZE * OPP_POKEMON_OBS_SIZE + # opp_bench_pokemon (padded) # ADDED
    # --- MODIFICATION END ---
    # Global State
    1 +                         # turn_number
    1 +                         # can_attach_energy_this_turn (0 or 1)
    1                           # is_first_turn (0 or 1)
)


class PokemonTCGPocketEnv(gym.Env):
    """
    Gymnasium environment for Pokemon TCG Pocket, modified for self-play training
    against historical opponents.
    """
    metadata = {'render_modes': ['human', 'text'], 'render_fps': 1}

    def __init__(self, render_mode: Optional[str] = None,
                 opponent_checkpoints_dir: str = "models/ppo_checkpoints", # Dir to load opponents from
                 opponent_pool_size: int = 5, # How many recent checkpoints to sample from (-1 for all)
                 always_load_latest_opponent: bool = False # For debugging: always use latest ckpt
                 ):
        super().__init__()
        self.render_mode = render_mode
        # --- MODIFICATION: Store opponent config ---
        self.opponent_checkpoints_dir = opponent_checkpoints_dir
        self.opponent_pool_size = opponent_pool_size
        self.always_load_latest_opponent = always_load_latest_opponent
        self._opponent_policy: Optional[BasePolicy] = None # Loaded policy network
        self._opponent_vec_normalize_stats: Optional[str] = None # Path to vecnormalize stats for the opponent model

        # --- Observation/Action Spaces (remain the same, using updated FLATTENED_OBS_SIZE) ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FLATTENED_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_POSSIBLE_ACTIONS)

        self.game: Optional[Game] = None
        # --- MODIFICATION: Distinguish learning agent player ---
        self.agent_player_idx: Optional[int] = None # 0 or 1, who is the learning agent?
        self.agent_player: Optional[Player] = None # Reference to agent's Player object
        self.opponent_player: Optional[Player] = None # Reference to opponent's Player object

        print(f"Environment Initialized (Self-Play Mode):")
        print(f"  Opponent Checkpoint Dir: {self.opponent_checkpoints_dir}")
        print(f"  Opponent Pool Size: {self.opponent_pool_size}")
        # Reference to the player whose turn it currently is within the game state.
        # Updated during reset() and step().
        self.current_player: Optional[Player] = None
        self.manual_override_enabled = False
        self.manual_state_queue = []


        print(f"Environment Initialized:")
        print(f"  Observation Space Shape: {self.observation_space.shape}")
        print(f"  Action Space Size: {self.action_space.n}")
        print(f"  Max Deck Size: {MAX_DECK_SIZE}")
        print(f"  Max Hand Size: {MAX_HAND_SIZE}")
        print(f"  Max Bench Size: {MAX_BENCH_SIZE}")
        print(f"  Max Attacks per Pokemon (Env Assumption): {MAX_ATTACKS_PER_POKEMON}")
        print(f"  Points to Win: {POINTS_TO_WIN}")

    def _find_opponent_checkpoints(self) -> List[str]:
        """Finds saved model checkpoints in the specified directory."""
        if not os.path.isdir(self.opponent_checkpoints_dir):
            return []
        # Find files matching the pattern (e.g., ppo_opponent_*_steps.zip)
        pattern = os.path.join(self.opponent_checkpoints_dir, "ppo_opponent_*_steps.zip")
        checkpoints = glob.glob(pattern)
        # Sort by step number (extracted from filename) to get recent ones
        checkpoints.sort(key=lambda f: int(f.split('_')[-2]), reverse=True)
        return checkpoints
    
    def _load_opponent_policy(self) -> bool:
        """Loads a policy for the opponent from a checkpoint file."""
        # Use a temporary variable to store the status message/name
        current_opponent_status = "None Loaded"

        if MaskablePPO is None: # Check if import failed
             print("ERROR: Cannot load opponent policy, Stable Baselines 3 not installed.")
             current_opponent_status = "SB3 Error"
             self._opponent_policy = None; self._opponent_vec_normalize_stats = None # Ensure reset
             self.loaded_opponent_basename = current_opponent_status # Update status
             return False

        checkpoints = self._find_opponent_checkpoints()
        if not checkpoints:
            if self.render_mode: print("No opponent checkpoints found. Opponent will play randomly (or use fallback).")
            current_opponent_status = "Random/Fallback" # Set status here
            self._opponent_policy = None
            self._opponent_vec_normalize_stats = None
            self.loaded_opponent_basename = current_opponent_status # Update status
            return False # Indicate no policy loaded

        # Select checkpoint (logic remains the same)
        if self.always_load_latest_opponent:
            selected_checkpoint_path = checkpoints[0]
        else:
            pool_size = self.opponent_pool_size if self.opponent_pool_size > 0 else len(checkpoints)
            pool = checkpoints[:min(pool_size, len(checkpoints))]
            selected_checkpoint_path = random.choice(pool)

        stats_path = selected_checkpoint_path.replace(".zip", "_vecnormalize.pkl")

        try:
            if self.render_mode: print(f"Loading opponent policy from: {selected_checkpoint_path}")
            loaded_model = MaskablePPO.load(selected_checkpoint_path, device='cpu')
            self._opponent_policy = loaded_model.policy
            self._opponent_policy.set_training_mode(False)
            current_opponent_status = os.path.basename(selected_checkpoint_path) # Set status on success

            if os.path.exists(stats_path):
                 self._opponent_vec_normalize_stats = stats_path
                 if self.render_mode: print(f"Found opponent VecNormalize stats: {stats_path}")
            else:
                 self._opponent_vec_normalize_stats = None
                 if self.render_mode: print(f"Warning: VecNormalize stats not found at {stats_path}.")

            self.loaded_opponent_basename = current_opponent_status # Update status
            return True # Policy loaded successfully

        except Exception as e:
            print(f"ERROR loading opponent policy from {selected_checkpoint_path}: {e}")
            current_opponent_status = f"ERROR loading {os.path.basename(selected_checkpoint_path)}" # Set error status
            traceback.print_exc()
            self._opponent_policy = None
            self._opponent_vec_normalize_stats = None
            self.loaded_opponent_basename = current_opponent_status # Update status
            return False

    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary information."""
        # Keep this simple - episode info added on termination
        info = {
             "agent_player_idx": self.agent_player_idx,
             "turn_number": self.game.game_state.turn_number if self.game else 0,
             "loaded_opponent": self.loaded_opponent_basename
        }
        return info


    def _setup_new_game(self):
        """
        Creates and initializes a new game instance using example decks.
        Called by reset().
        """
        deck1_names = ["Darkrai ex"] * 2 + ["Giratina ex"] * 2 + ["Professor's Research"] * 2 + ["Poké Ball"] * 2 + ["Potion"] * 2 + ["Giant Cape"] + ["Rocky Helmet"] * 2 + ["Sabrina"] + ["Leaf"] + ["Mars"] * 2 + ["Red"] + ["Cyrus"] + ["Pokémon Center Lady"]

        deck2_names = list(ARCHETYPE_CORE) # Make a copy

        # Randomly select tech cards to fill remaining slots
        available_tech = list(TECH_POOL.items())
        random.shuffle(available_tech)

        current_tech_count = 0
        for tech_card, max_count in available_tech:
            if current_tech_count < NUM_TECH_SLOTS:
                num_to_add = min(max_count, NUM_TECH_SLOTS - current_tech_count)
                if random.random() < 0.7:
                    deck2_names.extend([tech_card] * num_to_add)
                    current_tech_count += num_to_add
            else:
                break

        # Ensure final deck size is correct (sanity check)
        deck2_names = deck2_names[:MAX_DECK_SIZE]

        # Shuffle names to randomize deck order
        random.shuffle(deck1_names)
        random.shuffle(deck2_names)

        # Ensure decks have exactly MAX_DECK_SIZE names (Game also validates)
        deck1_names = deck1_names[:MAX_DECK_SIZE]
        deck2_names = deck2_names[:MAX_DECK_SIZE]

        # Provide the corresponding energy types for the energy stand mechanic
        player1_energy_types = ["Darkness"]
        player2_energy_types = ["Darkness"]

        try:
            self.game = Game(
                player1_deck_names=deck1_names, # Corrected parameter name
                player2_deck_names=deck2_names, # Corrected parameter name
                player1_energy_types=player1_energy_types,
                player2_energy_types=player2_energy_types
            )
            self.current_player = self.game.game_state.get_current_player()
            if self.render_mode == "human":
                print("New game started successfully.")
        except Exception as e:
            print(f"FATAL ERROR during game initialization: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize the game simulation.") from e


    def _get_pokemon_state_dict(self, pokemon: Optional[PokemonCard]) -> Optional[Dict[str, Any]]:
        """
        Extracts relevant data from a PokemonCard object into a dictionary format.
        Returns None if the input pokemon is None.
        """
        if pokemon is None or pokemon.is_fainted: # Treat fainted Pokemon as non-existent for observation
            return None

        return {
            "name": pokemon.name,
            "current_hp": pokemon.current_hp,
            "max_hp": pokemon.hp,
            "energy": pokemon.attached_energy.copy(), # Use a copy
            "is_ex": pokemon.is_ex,
            "type": pokemon.pokemon_type,
            "weakness": pokemon.weakness_type,
            "can_attack": [pokemon.can_attack(i) for i in range(len(pokemon.attacks))],
            "num_attacks": len(pokemon.attacks)
        }

    def _format_pokemon_obs(self, pokemon_state: Optional[Dict[str, Any]],
                         include_can_attack=True,
                         can_attack_flags_input: Optional[List[bool]] = None) -> np.ndarray:
        """
        Converts a Pokemon's state dictionary into a flat NumPy array segment
        for the observation vector. Handles padding and default values for non-existent Pokemon.
        """
        target_size = POKEMON_OBS_SIZE if include_can_attack else OPP_POKEMON_OBS_SIZE

        if pokemon_state is None:
            # Return a zero vector representing a non-existent/empty Pokemon slot
            return np.zeros(target_size, dtype=np.float32)

        # --- Encode Pokemon Data ---
        card_id = float(CARD_TO_ID.get(pokemon_state["name"], UNKNOWN_CARD_ID))
        current_hp = float(pokemon_state["current_hp"])
        max_hp = float(pokemon_state.get("hp", 0))

        # Encode attached energy into a fixed-size vector
        energy_counts = np.zeros(NUM_ENERGY_TYPES, dtype=np.float32)
        energy_dict = pokemon_state.get("attached_energy", {})
        for type_str, count in energy_dict.items():
            type_id = ENERGY_TO_ID.get(type_str, NO_ENERGY_ID) # Map unknown types to NO_ENERGY_ID for safety
            if 0 <= type_id < NUM_ENERGY_TYPES: # Ensure ID is valid before indexing
                 energy_counts[type_id] += float(count)

        is_ex = 1.0 if pokemon_state["is_ex"] else 0.0
        type_id = float(ENERGY_TO_ID.get(pokemon_state["pokemon_type"], NO_ENERGY_ID))
        weakness_id = float(ENERGY_TO_ID.get(pokemon_state["weakness_type"], NO_ENERGY_ID)) # NO_ENERGY_ID represents no weakness

        # Base observation components
        obs_list = [
            1.0, # exists flag
            card_id,
            current_hp,
            max_hp,
            *energy_counts, # Unpack the energy array
            is_ex,
            type_id,
            weakness_id,
        ]

        # Add 'can_attack' flags if required for this Pokemon's observation
        if include_can_attack:
            can_attack_flags = np.zeros(MAX_ATTACKS_PER_POKEMON, dtype=np.float32)
            num_actual_attacks = pokemon_state.get("num_attacks", 0)
            can_attack_list = pokemon_state.get("can_attack", [])
            for i in range(min(num_actual_attacks, MAX_ATTACKS_PER_POKEMON)):
                 if i < len(can_attack_list) and can_attack_list[i]:
                    can_attack_flags[i] = 1.0
            obs_list.extend(can_attack_flags)

        # Convert to NumPy array and verify size
        obs_array = np.array(obs_list, dtype=np.float32)
        if obs_array.shape[0] != target_size:
            # This should not happen if calculations are correct, but good safeguard
            raise ValueError(f"Pokemon observation size mismatch! Expected {target_size}, "
                             f"got {obs_array.shape[0]} for "
                             f"{pokemon_state.get('name', 'Unknown Pokemon')}. "
                             f"Check _calculate_pokemon_obs_size and _format_pokemon_obs.")

        return obs_array


    def _get_obs(self, perspective_player: Player) -> np.ndarray:
        """
        Constructs the flattened observation vector from the perspective of the given player.
        """
        if self.game is None:
            return np.zeros(FLATTENED_OBS_SIZE, dtype=np.float32)

        # Determine opponent relative to the perspective player
        opponent = self.game.player1 if perspective_player == self.game.player2 else self.game.player2

        # Get the game state dictionary from the game's perspective function
        state_dict = self.game.get_state_representation(perspective_player)

        # --- My (Perspective Player) State ---
        my_hand_ids = [CARD_TO_ID.get(name, UNKNOWN_CARD_ID) for name in state_dict["my_hand_cards"]]
        my_hand_padded = np.pad(my_hand_ids, (0, MAX_HAND_SIZE - len(my_hand_ids)),
                                constant_values=UNKNOWN_CARD_ID).astype(np.float32)
        my_deck_size = np.array([state_dict["my_deck_size"]], dtype=np.float32)
        my_discard_size = np.array([state_dict["my_discard_size"]], dtype=np.float32)
        my_points = np.array([state_dict["my_points"]], dtype=np.float32)
        my_energy_available_id = np.array([ENERGY_TO_ID.get(state_dict["my_energy_stand_available"], NO_ENERGY_ID)], dtype=np.float32)
        my_energy_preview_id = np.array([ENERGY_TO_ID.get(state_dict["my_energy_stand_preview"], NO_ENERGY_ID)], dtype=np.float32)

        # My Active Pokemon
        my_active_pokemon_obj = perspective_player.active_pokemon
        my_active_state_dict = state_dict["my_active_pokemon"]
        can_attack_flags_active = None
        if my_active_state_dict and my_active_pokemon_obj:
            can_attack_flags_active = [my_active_pokemon_obj.can_attack(i) for i in range(len(my_active_pokemon_obj.attacks))]
        my_active_flat = self._format_pokemon_obs(my_active_state_dict, include_can_attack=True, can_attack_flags_input=can_attack_flags_active)

        # My Bench Pokemon
        my_bench_flat_list = []
        my_bench_pokemon_dicts = state_dict["my_bench_pokemon"]
        for i in range(MAX_BENCH_SIZE):
            pokemon_obj = perspective_player.bench[i] if i < len(perspective_player.bench) else None
            bench_poke_state_dict = my_bench_pokemon_dicts[i] if i < len(my_bench_pokemon_dicts) else None
            can_attack_flags_bench = None
            if bench_poke_state_dict and pokemon_obj:
                can_attack_flags_bench = [pokemon_obj.can_attack(i) for i in range(len(pokemon_obj.attacks))]
            my_bench_flat_list.append(self._format_pokemon_obs(bench_poke_state_dict, include_can_attack=True, can_attack_flags_input=can_attack_flags_bench))
        my_bench_flat = np.concatenate(my_bench_flat_list)

        # --- Opponent State ---
        opp_hand_size = np.array([state_dict["opp_hand_size"]], dtype=np.float32)
        opp_deck_size = np.array([state_dict["opp_deck_size"]], dtype=np.float32)
        opp_discard_size = np.array([state_dict["opp_discard_size"]], dtype=np.float32)
        opp_points = np.array([state_dict["opp_points"]], dtype=np.float32)
        opp_energy_available_exists = np.array([1.0 if state_dict["opp_energy_stand_status"]["available_exists"] else 0.0], dtype=np.float32)
        opp_energy_preview_id = np.array([ENERGY_TO_ID.get(state_dict["opp_energy_stand_status"]["preview"], NO_ENERGY_ID)], dtype=np.float32)

        # Opponent Active Pokemon
        opp_active_state_dict = state_dict["opp_active_pokemon"]
        opp_active_flat = self._format_pokemon_obs(opp_active_state_dict, include_can_attack=False)

        # Opponent Bench Pokemon
        opp_bench_flat_list = []
        opp_bench_pokemon_dicts = state_dict["opp_bench_pokemon"]
        for i in range(MAX_BENCH_SIZE):
            opp_poke_state_dict = opp_bench_pokemon_dicts[i] if i < len(opp_bench_pokemon_dicts) else None
            opp_bench_flat_list.append(self._format_pokemon_obs(opp_poke_state_dict, include_can_attack=False))
        opp_bench_flat = np.concatenate(opp_bench_flat_list)

        # --- Global State ---
        turn_number = np.array([state_dict["turn"]], dtype=np.float32)
        can_attach_energy = np.array([1.0 if state_dict["can_attach_energy"] else 0.0], dtype=np.float32)
        is_first_turn = np.array([1.0 if state_dict["is_first_turn"] else 0.0], dtype=np.float32)

        # --- Concatenate All Parts ---
        flat_obs = np.concatenate([
            my_hand_padded, my_deck_size, my_discard_size, my_points,
            my_energy_available_id, my_energy_preview_id, my_active_flat, my_bench_flat,
            opp_hand_size, opp_deck_size, opp_discard_size, opp_points,
            opp_energy_available_exists, opp_energy_preview_id, opp_active_flat, opp_bench_flat,
            turn_number, can_attach_energy, is_first_turn,
        ])

        if flat_obs.shape[0] != FLATTENED_OBS_SIZE:
             raise ValueError(f"Observation size mismatch! Expected {FLATTENED_OBS_SIZE}, got {flat_obs.shape[0]}")

        return flat_obs.astype(np.float32)

    def action_mask_fn(self, perspective_player: Player) -> np.ndarray:
        """Computes the action mask for the given player's perspective."""
        if self.game is None:
             return np.zeros(NUM_POSSIBLE_ACTIONS, dtype=bool)

        # Temporarily set game context for get_possible_actions (if it depends on current_player)
        original_player_index = self.game.game_state.current_player_index
        target_player_index = 0 if perspective_player == self.game.player1 else 1
        self.game.game_state.current_player_index = target_player_index

        possible_actions_str = self.game.get_possible_actions()
        valid_mask = np.zeros(NUM_POSSIBLE_ACTIONS, dtype=bool)

        # Restore original player index
        self.game.game_state.current_player_index = original_player_index

        # --- Mapping logic remains the same ---
        for action_str in possible_actions_str:
            action_id = ACTION_MAP.get(action_str) # Simplified mapping check

            # Handle parameterized actions if direct map failed
            if action_id is None:
                parts = action_str.split('_')
                if len(parts) > 1:
                    # Check common prefixes (add more as needed)
                    if action_str.startswith(ACTION_ATTACH_ENERGY_BENCH_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_PLAY_BASIC_BENCH_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_ATTACK_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_USE_ABILITY_BENCH_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_PLAY_SUPPORTER_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_PLAY_ITEM_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_ATTACH_TOOL_ACTIVE): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_ATTACH_TOOL_BENCH_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_RETREAT_TO_BENCH_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX): action_id = ACTION_MAP.get(action_str)
                    elif action_str.startswith(ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX): action_id = ACTION_MAP.get(action_str)
                    # Add checks for targetted trainer actions if needed

            if action_id is not None and 0 <= action_id < NUM_POSSIBLE_ACTIONS:
                valid_mask[action_id] = True

        if not np.any(valid_mask):
            pass_action_id = ACTION_MAP.get(ACTION_PASS)
            if pass_action_id is not None and 0 <= pass_action_id < NUM_POSSIBLE_ACTIONS:
                 valid_mask[pass_action_id] = True
            else:
                 print("FATAL ERROR: No valid actions possible for mask, and PASS is invalid!")
        return valid_mask

    def step(self, action: int):
        if self.game is None or self.agent_player is None or self.opponent_player is None:
             raise RuntimeError("step() called before reset().")

        terminated = False
        truncated = False
        reward = 0.0
        info = {} # Start with empty info for this step

        # --- Execute Agent's Action ---
        action_str = ID_TO_ACTION.get(action)
        if action_str is None:
            # ... (Handling for invalid action ID - remains the same) ...
            print(f"Warning: Agent provided invalid action ID {action}. Taking no action, assigning penalty.")
            reward = -1.0
            terminated = False
            info["error"] = f"Invalid action ID {action}"
        else:
            # --- Agent Action Execution Block ---
            if self.render_mode == "human":
                print(f"\n>>> Agent ({self.agent_player.name}) Action: {action_str} (ID: {action})")
            try:
                _next_state_dict, step_reward, game_terminated = self.game.step(action_str)
                reward += float(step_reward)
                terminated = game_terminated # Update terminated flag

                if terminated:
                     # Game ended after agent's action
                     observation = self._get_obs(self.agent_player)
                     info = self._get_info() # Get standard info
                     # --- Determine Winner ---
                     agent_won = 0.0
                     if self.agent_player.points >= POINTS_TO_WIN: agent_won = 1.0
                     elif self.opponent_player.points >= POINTS_TO_WIN: agent_won = 0.0
                     elif self.game.game_state.turn_number > self.game.turn_limit: agent_won = 0.5
                     # --- Populate TOP LEVEL Info for Monitor ---
                     info['w'] = agent_won # Add 'w' directly
                     # Monitor will add/use 'r', 'l', 't'
                     info['r'] = reward # Pass current step reward
                     info['l'] = self.game.game_state.turn_number # Pass current length
                     info['t'] = time.time() # Pass current time
                     info["final_observation"] = observation
                     info['TimeLimit.truncated'] = False
                     # --- End Monitor Info Population ---

                     if self.render_mode: self.render()
                     # print(f"[DEBUG ENV STEP EARLY RETURN] Terminated={terminated}, Info={info}") # Optional Debug
                     return observation, reward, terminated, truncated, info # Return immediately


            except Exception as e:
                # ... (Error handling for agent's step - remains the same) ...
                print(f"FATAL ERROR during game.step with AGENT action '{action_str}'")
                traceback.print_exc()
                observation = self._get_obs(self.agent_player); reward = -10.0; terminated = True; info = self._get_info(); info["error"] = f"Agent step exception: {e}"
                # Optionally add info['episode'] here too if you want to record loss on error
                # info['episode'] = {'w': 0.0, 'r': reward, 'l': self.game.game_state.turn_number}
                return observation, reward, terminated, truncated, info

        # --- Simulate Opponent's Turn(s) only if game didn't end on agent's turn ---
        opponent_accumulated_reward = 0.0
        if not terminated:
            try:
                while not terminated and self.game.game_state.current_player_index != self.agent_player_idx:
                    game_terminated_opp, reward_opp = self._simulate_opponent_turn()
                    opponent_accumulated_reward += reward_opp
                    terminated = game_terminated_opp # Update terminated flag

            except Exception as e:
                 # ... (Error handling for opponent simulation - remains the same) ...
                 print(f"ERROR during opponent turn simulation: {e}")
                 traceback.print_exc()
                 terminated = True # Assume game is broken
                 info["error"] = f"Opponent simulation exception: {e}"
                 # We will determine winner and add info['episode'] below

        # --- Get Final State for Agent ---
        observation = self._get_obs(self.agent_player)
        # Get standard info first, might be overwritten by episode info if terminated
        info = self._get_info()

        # --- ADDED BLOCK: Check termination *after* opponent simulation ---
        if terminated: # Check if episode info wasn't already set by agent's win
             if 'w' not in info: # Check if 'w' wasn't already added
                 # Determine winner based on state AFTER opponent's turn(s)
                 agent_won = 0.0
                 if self.agent_player.points >= POINTS_TO_WIN: agent_won = 1.0
                 elif self.opponent_player.points >= POINTS_TO_WIN: agent_won = 0.0
                 elif self.game.game_state.turn_number > self.game.turn_limit: agent_won = 0.5
                 # --- Populate TOP LEVEL Info for Monitor ---
                 info['w'] = agent_won # Add 'w' directly
                 info['r'] = reward # Use agent's reward from its step
                 info['l'] = self.game.game_state.turn_number
                 info['t'] = time.time()
                 info["final_observation"] = observation
                 info['TimeLimit.truncated'] = False
                 # --- End Monitor Info Population ---

        # --- END ADDED BLOCK ---

        # --- Rendering ---
        if self.render_mode == "human":
            self.render()
            print(f"<<< Step Result: Agent Reward={reward}, Terminated={terminated}")
        elif self.render_mode == "text":
            self._render_text(action_str, reward, terminated)

        return observation, reward, terminated, truncated, info
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._setup_new_game()
        self._load_opponent_policy()

        if self.game is None:
            raise RuntimeError("Game object not initialized after _setup_new_game.")

        self.agent_player_idx = random.choice([0, 1])
        if self.agent_player_idx == 0:
            self.agent_player = self.game.player1
            self.opponent_player = self.game.player2
        else:
            self.agent_player = self.game.player2
            self.opponent_player = self.game.player1

        print(f"Game Reset: Agent is Player {self.agent_player_idx + 1} ({self.agent_player.name})")

        initial_observation = None
        terminated = False # Flag if game ends during opponent's first turn
        info = {} # Initialize info dict

        try:
            while self.game.game_state.current_player_index != self.agent_player_idx:
                if self.render_mode == "human": print("Opponent starts first...")
                # Simulate opponent turn, get terminated status
                opponent_done, _ = self._simulate_opponent_turn()
                if opponent_done:
                    terminated = True
                    initial_observation = self._get_obs(self.agent_player)
                    # --- Determine Winner and Populate TOP LEVEL Info ---
                    agent_won = 0.0
                    if self.agent_player and self.opponent_player:
                        if self.opponent_player.points >= POINTS_TO_WIN: agent_won = 0.0
                        elif self.agent_player.points >= POINTS_TO_WIN: agent_won = 1.0
                        elif self.game.game_state.turn_number > self.game.turn_limit: agent_won = 0.5

                    # Add 'w' directly to info dict
                    info['w'] = agent_won
                    # Monitor also needs 'r', 'l', 't' potentially (add defaults)
                    info['r'] = 0.0 # No reward for agent yet
                    info['l'] = self.game.game_state.turn_number
                    info['t'] = time.time() # Monitor uses time

                    # Store final observation for SB3 >= v2.10 / Gymnasium >= v0.26
                    info["final_observation"] = initial_observation
                    info['TimeLimit.truncated'] = False
                    break # Exit loop
        except Exception as e:
             print(f"ERROR during initial opponent turn simulation: {e}")
             traceback.print_exc()
             initial_observation = self._get_obs(self.agent_player)
             # If error, maybe don't populate 'episode' info, let next step handle termination?

        if initial_observation is None:
            initial_observation = self._get_obs(self.agent_player)

        # If terminated during reset, info is already populated
        # Otherwise, get standard info
        if not terminated:
             info = self._get_info()

        final_obs = initial_observation
        final_info = info if info is not None else {}

        if self.render_mode == "human": self.render()
        # print(f"[DEBUG ENV RESET RETURN] Terminated={terminated}, Info={final_info}") # Optional Debug
        return final_obs, final_info
    
    def _simulate_opponent_turn(self) -> Tuple[bool, float]:
        """
        Simulates the opponent's entire turn using the loaded policy.
        Returns (terminated, accumulated_reward_for_opponent).
        Reward is from opponent's perspective, may not be directly useful for agent.
        """
        if self.game is None or self.opponent_player is None: return True, 0.0 # Game ended state

        terminated = False
        accumulated_reward = 0.0
        turn_continues = True

        while turn_continues and not terminated and self.game.game_state.current_player_index != self.agent_player_idx:
            if self._opponent_policy:
                # Get opponent observation and mask
                obs_opp = self._get_obs(self.opponent_player)
                mask_opp = self.action_mask_fn(self.opponent_player)

                # Normalize observation for opponent policy
                # --- Normalization Handling ---
                # Option A: Use stats saved with the opponent model (more accurate)
                # Requires loading VecNormalize object based on self._opponent_vec_normalize_stats
                # This adds complexity with managing DummyVecEnv/VecNormalize loading here.
                # Option B: Use the *current* VecNormalize stats (simpler, less accurate if stats drift)
                # To do Option B, we need access to the current VecNormalize wrapper applied *outside* this env.
                # ---> Let's SKIP normalization here for simplicity first. Assume normalized env or handle outside.
                # normalized_obs_opp = ???

                # Predict action using opponent policy
                try:
                    # IMPORTANT: Use MaskablePPO prediction if that's what was saved
                    action_id_opp, _ = self._opponent_policy.predict(
                        obs_opp.reshape(1, -1), # Reshape for batch dim
                        action_masks=mask_opp.reshape(1, -1),
                        deterministic=True # Or False for stochastic opponent
                    )
                    action_id_opp = int(action_id_opp.item()) # Extract scalar
                except Exception as e:
                     print(f"ERROR during opponent policy prediction: {e}")
                     traceback.print_exc()
                     action_id_opp = ACTION_MAP.get(ACTION_PASS) # Fallback to PASS on error
                     if action_id_opp is None: return True, accumulated_reward # Cannot even pass

            else:
                # No policy loaded - opponent plays randomly among valid actions
                mask_opp = self.action_mask_fn(self.opponent_player)
                valid_ids = np.where(mask_opp)[0]
                action_id_opp = random.choice(valid_ids) if len(valid_ids) > 0 else ACTION_MAP.get(ACTION_PASS)
                if action_id_opp is None: return True, accumulated_reward # Cannot even pass

            # Convert ID to string and execute in game
            action_str_opp = ID_TO_ACTION.get(action_id_opp)
            if action_str_opp is None:
                print(f"ERROR: Opponent chose invalid action ID {action_id_opp}. Forcing PASS.")
                action_str_opp = ACTION_PASS

            if self.render_mode == "human": print(f"--- Opponent ({self.opponent_player.name}) Action: {action_str_opp} (ID: {action_id_opp})")

            _next_state_dict, reward_opp, terminated = self.game.step(action_str_opp)
            accumulated_reward += reward_opp

            # Check if the opponent's action ended their turn (game state switched player)
            if self.game.game_state.current_player_index == self.agent_player_idx:
                turn_continues = False

            # Break loop if game ended
            if terminated:
                turn_continues = False

        return terminated, accumulated_reward


    def render(self):
        """Renders the environment based on the selected render_mode."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "text":
            # Text rendering might be done incrementally in step, or snapshot here
             pass # Already handled in step for text mode in this example


    def _render_human(self):
        """Provides a detailed text-based rendering for human observation."""
        if self.game is None:
            print("Cannot render: Game not initialized.")
            return
        # Ensure agent/opponent roles are set (should be after reset)
        if self.agent_player_idx is None or self.agent_player is None or self.opponent_player is None:
             print("Cannot render: Agent/Opponent roles not set (Env not reset properly?).")
             return

        print("\n" + "="*40)
        gs = self.game.game_state
        p1 = self.game.player1
        p2 = self.game.player2
        current_player_render = gs.get_current_player() # Player whose turn it is in the game

        # Determine labels based on who the learning agent is
        agent_label = f"Player {self.agent_player_idx + 1} ({self.agent_player.name}) (Agent)"
        opp_label = f"Player {1 - self.agent_player_idx + 1} ({self.opponent_player.name}) (Opponent)"
        p1_label = agent_label if self.agent_player_idx == 0 else opp_label
        p2_label = agent_label if self.agent_player_idx == 1 else opp_label

        print(f"Turn: {gs.turn_number} | Current Turn Player: {current_player_render.name} ({'Agent' if current_player_render == self.agent_player else 'Opponent'})")
        print(f"First Turn Overall? {'Yes' if gs.is_first_turn else 'No'}")
        print(f"Opponent Policy Loaded: {'Yes' if self._opponent_policy else 'No'} ({os.path.basename(self._opponent_vec_normalize_stats) if self._opponent_vec_normalize_stats else 'N/A'})")
        print("-" * 20)

        # --- Player 1 Info ---
        print(p1_label) # Use the dynamic label
        print(f"  Points: {p1.points}/{POINTS_TO_WIN}")
        print(f"  Hand: {len(p1.hand)} | Deck: {len(p1.deck)} | Discard: {len(p1.discard_pile)}")
        print(f"  Energy Stand: Avail='{p1.energy_stand_available or 'None'}', Preview='{p1.energy_stand_preview or 'None'}'")
        print(f"  Active: {p1.active_pokemon!r}") # Use repr for Pokemon details
        bench_str_p1 = ", ".join(repr(p) for p in p1.bench) if p1.bench else "Empty"
        print(f"  Bench ({len(p1.bench)}/{MAX_BENCH_SIZE}): [{bench_str_p1}]")
        print("-" * 20)

        # --- Player 2 Info ---
        print(p2_label) # Use the dynamic label
        print(f"  Points: {p2.points}/{POINTS_TO_WIN}")
        print(f"  Hand: {len(p2.hand)} | Deck: {len(p2.deck)} | Discard: {len(p2.discard_pile)}")
        print(f"  Energy Stand: Avail='{p2.energy_stand_available or 'None'}', Preview='{p2.energy_stand_preview or 'None'}'")
        print(f"  Active: {p2.active_pokemon!r}") # Use repr for Pokemon details
        bench_str_p2 = ", ".join(repr(p) for p in p2.bench) if p2.bench else "Empty"
        print(f"  Bench ({len(p2.bench)}/{MAX_BENCH_SIZE}): [{bench_str_p2}]")
        print("-" * 20)

        # --- Possible Actions & Mask Info ---
        # Show possible actions for the player whose turn it currently is in the game
        print(f"Possible Actions for Current Player ({current_player_render.name}):")
        try:
            # We need the actions available to the player whose turn it IS in the game state
            possible_actions = self.game.get_possible_actions() # This uses game.game_state.current_player_index
            for pa in possible_actions:
                print(f"  - {pa}")
        except Exception as e:
            print(f"\nError getting possible actions for rendering: {e}")

        # Display action mask, which is always relevant from the LEARNING AGENT's perspective
        print(f"\nAction Mask Info (Perspective: {self.agent_player.name} (Agent)):")
        try:
            # Get the mask from the agent's point of view
            mask = self.action_mask_fn(self.agent_player)
            valid_action_ids = np.where(mask)[0]
            valid_action_names = [ID_TO_ACTION.get(idx, f'INVALID_ID_{idx}') for idx in valid_action_ids]
            print(f"  Valid Action IDs (Agent): {valid_action_ids.tolist()}")
            print(f"  Valid Action Names (Agent): {valid_action_names}")
        except Exception as e:
             print(f"\nError generating action mask for rendering: {e}")

        print("="*40 + "\n")

    def _render_text(self, action_str: Optional[str], reward: float, terminated: bool):
         """ Minimal text rendering, often called from step. """
         if self.game is None or self.current_player is None: return
         current_player_name = self.current_player.name # Player *after* action
         print(f"T{self.game.game_state.turn_number}: {current_player_name} took '{action_str}', R={reward}, Done={terminated}")


    def close(self):
        """Clean up any resources (not strictly needed for this simple env)."""
        if self.render_mode == "human":
            print("Closing Pokemon TCG Pocket Environment.")
        self.game = None


# --- Example Usage Block ---
if __name__ == '__main__':
    print("Starting Environment Example Usage...")
    # Use render_mode="human" for detailed turn-by-turn output
    env = PokemonTCGPocketEnv(render_mode="human")

    try:
        obs, info = env.reset()
        print("\nInitial Observation Received (showing first 10 elements):")
        print(obs[:10])
        print(f"Initial Observation Shape: {obs.shape}")
        print(f"Initial Info: {info}")

        # Test a few steps with random valid actions using the action mask
        for i in range(15): # Run more steps for better testing
            print(f"\n--- Simulation Step {i+1} ---")

            # Get the action mask by calling the function directly
            # This simulates what the ActionMasker wrapper would do.
            current_mask = env.action_mask_fn()
            valid_action_ids = np.where(current_mask)[0]

            if env.render_mode == "human": # Already printed in render, but good for clarity here
                 print(f"Valid action IDs from mask: {valid_action_ids.tolist()}")
                 valid_action_names = [ID_TO_ACTION.get(idx, f'INVALID_ID_{idx}') for idx in valid_action_ids]
                 print(f"Valid action names from mask: {valid_action_names}")

            if len(valid_action_ids) > 0:
                # Choose a random valid action
                action = np.random.choice(valid_action_ids)
                action_name = ID_TO_ACTION.get(action, f'UNKNOWN_ID_{action}')
                print(f"----> Choosing valid random action: ID={action}, Name='{action_name}'")
            else:
                # This case indicates a potential issue in the game logic or mask generation
                print("ERROR: No valid actions found according to the mask! Something is wrong.")
                # Attempting to pass might be a fallback, but the root cause should be fixed.
                pass_action_id = ACTION_MAP.get(ACTION_PASS)
                if pass_action_id is not None:
                    print(f"Attempting PASS action (ID: {pass_action_id}) as fallback.")
                    action = pass_action_id
                else:
                    print("FATAL: PASS action not found in map. Cannot continue.")
                    break # Stop simulation

            # Perform the step
            obs, reward, terminated, truncated, info = env.step(action)

            # print(f"Reward: {reward}") # Already printed in human render/step
            # print(f"Terminated: {terminated}, Truncrated: {truncated}") # Already printed
            # print(f"New Observation (first 10): {obs[:10]}") # Can be verbose
            # print(f"New Info: {info}") # Already printed

            if terminated or truncated:
                print("\n======================")
                print(f"Episode finished after step {i+1}!")
                winning_player = info.get("winner", "N/A") # Assuming Game might add winner to info
                print(f"Winner: {winning_player}")
                print("Resetting environment...")
                print("======================\n")
                obs, info = env.reset()
                # No need to check mask here, next loop iteration will get it

    except Exception as e:
        print(f"\n--- An error occurred during the example run ---")
        print(e)
        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment Closed.")
