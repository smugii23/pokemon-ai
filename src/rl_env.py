# rl_env.py (Corrected and Refined)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List, Optional, Dict, Tuple, Any
import copy
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
    Gymnasium environment for the Pokemon TCG Pocket Simulator.

    Features:
    - Flattened observation space (suitable for standard MLP policies).
    - Discrete action space based on mapped game actions.
    - Action masking support via the `action_mask_fn` method (for use with SB3-Contrib's ActionMasker wrapper and MaskablePPO).
    """
    metadata = {'render_modes': ['human', 'text'], 'render_fps': 1}

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initializes the environment.

        Args:
            render_mode: The mode for rendering ('human', 'text', or None).
        """
        super().__init__()

        if render_mode not in (None, 'human', 'text'):
             print(f"Warning: Invalid render_mode '{render_mode}'. Defaulting to None.")
             self.render_mode = None
        else:
             self.render_mode = render_mode

        # --- Observation Space ---
        # A single flat vector of floats. Normalization (e.g., using VecNormalize)
        # is recommended during training as values have different scales (HP vs flags).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FLATTENED_OBS_SIZE,), dtype=np.float32
        )

        # --- Action Space ---
        # A single integer representing one of the possible mapped actions.
        self.action_space = spaces.Discrete(NUM_POSSIBLE_ACTIONS)

        # --- Game Instance ---
        # The core simulator object. Initialized in reset().
        self.game: Optional[Game] = None
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

    def _format_pokemon_obs(self, pokemon_state: Optional[Dict[str, Any]], include_can_attack=True) -> np.ndarray:
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


    def _get_obs(self) -> np.ndarray:
        """
        Constructs the flattened observation vector from the current game state
        from the perspective of the `self.current_player`. Includes opponent bench details.
        """
        if self.game is None or self.current_player is None:
            print("Warning: _get_obs called before game reset. Returning zeros.")
            return np.zeros(FLATTENED_OBS_SIZE, dtype=np.float32)

        state_dict = self.game.get_state_representation(self.current_player)
        opponent = self.game.game_state.get_opponent()

        # --- My (Current Player) State ---
        my_hand_ids = [CARD_TO_ID.get(name, UNKNOWN_CARD_ID) for name in state_dict["my_hand_cards"]]
        my_hand_padded = np.pad(my_hand_ids, (0, MAX_HAND_SIZE - len(my_hand_ids)),
                                constant_values=UNKNOWN_CARD_ID).astype(np.float32)
        my_deck_size = np.array([state_dict["my_deck_size"]], dtype=np.float32)
        my_discard_size = np.array([state_dict["my_discard_size"]], dtype=np.float32)
        my_points = np.array([state_dict["my_points"]], dtype=np.float32)
        my_energy_available_id = np.array([ENERGY_TO_ID.get(state_dict["my_energy_stand_available"], NO_ENERGY_ID)], dtype=np.float32)
        my_energy_preview_id = np.array([ENERGY_TO_ID.get(state_dict["my_energy_stand_preview"], NO_ENERGY_ID)], dtype=np.float32)

        # My Active Pokemon
        my_active_pokemon_obj = self.current_player.active_pokemon
        my_active_state_dict = state_dict["my_active_pokemon"] # Get dict from game state
        if my_active_state_dict and my_active_pokemon_obj: # Add calculated can_attack flags
            my_active_state_dict["can_attack_flags_calculated"] = [my_active_pokemon_obj.can_attack(i) for i in range(len(my_active_pokemon_obj.attacks))]
        my_active_flat = self._format_pokemon_obs(my_active_state_dict, include_can_attack=True)

        # My Bench Pokemon
        my_bench_flat_list = []
        my_bench_pokemon_dicts = state_dict["my_bench_pokemon"] # List of dicts/None
        for i in range(MAX_BENCH_SIZE):
            pokemon_obj = self.current_player.bench[i] if i < len(self.current_player.bench) else None
            bench_poke_state_dict = my_bench_pokemon_dicts[i] if i < len(my_bench_pokemon_dicts) else None # Get dict
            if bench_poke_state_dict and pokemon_obj: # Add calculated can_attack flags
                bench_poke_state_dict["can_attack_flags_calculated"] = [pokemon_obj.can_attack(i) for i in range(len(pokemon_obj.attacks))]
            my_bench_flat_list.append(self._format_pokemon_obs(bench_poke_state_dict, include_can_attack=True))
        my_bench_flat = np.concatenate(my_bench_flat_list)

        # --- Opponent State ---
        opp_hand_size = np.array([state_dict["opp_hand_size"]], dtype=np.float32)
        opp_deck_size = np.array([state_dict["opp_deck_size"]], dtype=np.float32)
        opp_discard_size = np.array([state_dict["opp_discard_size"]], dtype=np.float32)
        opp_points = np.array([state_dict["opp_points"]], dtype=np.float32)
        opp_energy_available_exists = np.array([1.0 if state_dict["opp_energy_stand_status"]["available_exists"] else 0.0], dtype=np.float32)
        opp_energy_preview_id = np.array([ENERGY_TO_ID.get(state_dict["opp_energy_stand_status"]["preview"], NO_ENERGY_ID)], dtype=np.float32)

        # Opponent Active Pokemon
        opp_active_state_dict = state_dict["opp_active_pokemon"] # Get dict
        opp_active_flat = self._format_pokemon_obs(opp_active_state_dict, include_can_attack=False)

        # --- MODIFICATION START: Opponent Bench Pokemon ---
        opp_bench_flat_list = []
        opp_bench_pokemon_dicts = state_dict["opp_bench_pokemon"] # Get list of dicts/None from game state
        for i in range(MAX_BENCH_SIZE):
            # Get the dict for the current slot, handle if bench isn't full
            opp_poke_state_dict = opp_bench_pokemon_dicts[i] if i < len(opp_bench_pokemon_dicts) else None
            # Format using OPP_POKEMON_OBS_SIZE (no can_attack needed)
            opp_bench_flat_list.append(self._format_pokemon_obs(opp_poke_state_dict, include_can_attack=False))
        opp_bench_flat = np.concatenate(opp_bench_flat_list) # Concatenate the formatted bench slots
        # --- MODIFICATION END ---

        # --- Global State ---
        turn_number = np.array([state_dict["turn"]], dtype=np.float32)
        can_attach_energy = np.array([1.0 if state_dict["can_attach_energy"] else 0.0], dtype=np.float32)
        is_first_turn = np.array([1.0 if state_dict["is_first_turn"] else 0.0], dtype=np.float32)

        # --- Concatenate All Parts (MODIFIED ORDER) ---
        flat_obs = np.concatenate([
            my_hand_padded,
            my_deck_size, my_discard_size, my_points,
            my_energy_available_id, my_energy_preview_id,
            my_active_flat,
            my_bench_flat,
            opp_hand_size, opp_deck_size, opp_discard_size, opp_points,
            opp_energy_available_exists, opp_energy_preview_id,
            opp_active_flat,
            # --- MODIFICATION START ---
            # Replace opp_bench_size with opp_bench_flat
            # opp_bench_size, # REMOVED
            opp_bench_flat, # ADDED
            # --- MODIFICATION END ---
            turn_number,
            can_attach_energy,
            is_first_turn,
        ])

        # Final size validation check
        if flat_obs.shape[0] != FLATTENED_OBS_SIZE:
            print(f"FATAL: Observation size mismatch! Expected {FLATTENED_OBS_SIZE}, got {flat_obs.shape[0]}. Check calculation and concatenation order.")
            # Print details of components to help debug (Update component sizes)
            print(f"  my_hand_padded: {my_hand_padded.shape}")          # MAX_HAND_SIZE
            print(f"  my nums: 3")                                      # 3
            print(f"  my energy: 2")                                    # 2
            print(f"  my_active_flat: {my_active_flat.shape}")          # POKEMON_OBS_SIZE
            print(f"  my_bench_flat: {my_bench_flat.shape}")            # MAX_BENCH_SIZE * POKEMON_OBS_SIZE
            print(f"  opp nums: 4")                                      # 4
            print(f"  opp energy: 2")                                    # 2
            print(f"  opp_active_flat: {opp_active_flat.shape}")        # OPP_POKEMON_OBS_SIZE
            print(f"  opp_bench_flat: {opp_bench_flat.shape}")          # MAX_BENCH_SIZE * OPP_POKEMON_OBS_SIZE
            print(f"  global: 3")                                      # 3

            total_components = (
                my_hand_padded.shape[0] + 3 + 2 +
                my_active_flat.shape[0] + my_bench_flat.shape[0] +
                4 + 2 + opp_active_flat.shape[0] + opp_bench_flat.shape[0] + 3
            )
            print(f"  Sum of component sizes: {total_components}")
            raise ValueError("Observation size mismatch detected.")

        return flat_obs.astype(np.float32)

    def action_mask_fn(self) -> np.ndarray:
        """
        Computes the action mask for the current game state.

        Returns:
            A boolean NumPy array of shape (NUM_POSSIBLE_ACTIONS,) where True
            indicates a valid action and False indicates an invalid action.
            This is used by the ActionMasker wrapper.
        """
        if self.game is None or self.current_player is None:
            # Should not happen in normal flow, return a mask allowing nothing
             print("Warning: action_mask_fn called before game reset. Allowing no actions.")
             return np.zeros(NUM_POSSIBLE_ACTIONS, dtype=bool)

        # Get the list of valid action strings from the game simulator
        possible_actions_str = self.game.get_possible_actions()
        valid_mask = np.zeros(NUM_POSSIBLE_ACTIONS, dtype=bool) # Initialize mask with all False
    

        # Iterate through the valid action strings and map them to their integer IDs
        for action_str in possible_actions_str:
            action_id = -1 # Default to invalid ID

            # --- Map action string to action ID ---
            # Check direct mapping first (covers PASS, ATTACH_ENERGY_ACTIVE, USE_ABILITY_ACTIVE)
            if action_str in ACTION_MAP:
                action_id = ACTION_MAP[action_str]
            else:
                # Handle parameterized actions (e.g., "play_basic_bench_3", "attack_0")
                parts = action_str.split('_')
                prefix = "_".join(parts[:-1]) + "_" # Reconstruct prefix like "play_basic_bench_"
                try:
                    index = int(parts[-1])
                    # Reconstruct the potential key in ACTION_MAP based on prefix and index range
                    map_key = None
                    if prefix == ACTION_PLAY_BASIC_BENCH_PREFIX and 0 <= index < MAX_HAND_SIZE:
                        map_key = f"{ACTION_PLAY_BASIC_BENCH_PREFIX}{index}"
                    elif prefix == ACTION_ATTACK_PREFIX and 0 <= index < MAX_ATTACKS_PER_POKEMON:
                        map_key = f"{ACTION_ATTACK_PREFIX}{index}"
                    elif prefix == ACTION_ATTACH_ENERGY_BENCH_PREFIX and 0 <= index < MAX_BENCH_SIZE:
                         map_key = f"{ACTION_ATTACH_ENERGY_BENCH_PREFIX}{index}"
                    elif prefix == ACTION_PLAY_SUPPORTER_PREFIX and 0 <= index < MAX_HAND_SIZE:
                         map_key = f"{ACTION_PLAY_SUPPORTER_PREFIX}{index}"
                    elif prefix == ACTION_PLAY_ITEM_PREFIX and 0 <= index < MAX_HAND_SIZE:
                         map_key = f"{ACTION_PLAY_ITEM_PREFIX}{index}"
                    elif prefix == ACTION_ATTACH_TOOL_ACTIVE and 0 <= index < MAX_HAND_SIZE:
                         map_key = f"{ACTION_ATTACH_TOOL_ACTIVE}{index}"
                    elif prefix == ACTION_RETREAT_TO_BENCH_PREFIX and 0 <= index < MAX_BENCH_SIZE:
                         map_key = f"{ACTION_RETREAT_TO_BENCH_PREFIX}{index}"
                    elif prefix == ACTION_USE_ABILITY_BENCH_PREFIX and 0 <= index < MAX_BENCH_SIZE: # <-- ADD THIS BLOCK
                        map_key = f"{ACTION_USE_ABILITY_BENCH_PREFIX}{index}"
                    elif prefix == ACTION_PLAY_SUPPORTER_PREFIX and 0 <= index < MAX_HAND_SIZE:
                        map_key = f"{ACTION_PLAY_SUPPORTER_PREFIX}{index}"
                    elif prefix == ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX and 0 <= index < MAX_HAND_SIZE:
                         map_key = f"{ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX}{index}"
                    elif prefix == ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX and 0 <= index < MAX_HAND_SIZE:
                         map_key = f"{ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX}{index}"
                    # Note: ACTION_SETUP_CONFIRM_READY is handled by the direct mapping check earlier

                    # Handle compound keys like ATTACH_TOOL_BENCH_BenchIndex_HandIndex
                    elif prefix == ACTION_ATTACH_TOOL_BENCH_PREFIX:
                         try:
                             # Expecting format like "ATTACH_TOOL_BENCH_0_5"
                             bench_idx_str = parts[-2]
                             hand_idx_str = parts[-1]
                             bench_idx = int(bench_idx_str)
                             hand_idx = int(hand_idx_str)
                             if 0 <= bench_idx < MAX_BENCH_SIZE and 0 <= hand_idx < MAX_HAND_SIZE:
                                 map_key = f"{ACTION_ATTACH_TOOL_BENCH_PREFIX}{bench_idx}_{hand_idx}"
                         except (ValueError, IndexError):
                              pass # Invalid format for this prefix

                    if map_key and map_key in ACTION_MAP:
                        action_id = ACTION_MAP[map_key]
                    # else:
                    #     # This might happen if game yields an action we haven't mapped
                    #     # or index is out of range for the mapping (e.g., hand size changes unexpectedly)
                    #     if self.render_mode: print(f"Warning: Could not map parameterized action '{action_str}' to ID.")

                except (ValueError, IndexError):
                    # Handle cases where the last part isn't an integer or action_str is malformed
                    # if self.render_mode: print(f"Warning: Error parsing action string '{action_str}'.")
                    pass # Keep action_id as -1

            # --- Update the mask ---
            # If a valid action_id was found and is within the expected range, mark it as True
            if action_id != -1 and 0 <= action_id < NUM_POSSIBLE_ACTIONS:
                valid_mask[action_id] = True
            # else:
            #     # This might indicate an issue with ACTION_MAP or get_possible_actions()
            #     if action_str: # Don't warn if action_str was empty/None
            #         if self.render_mode: print(f"Warning: Action string '{action_str}' generated invalid ID {action_id}.")


        # Ensure at least one action is always possible (usually PASS should be)
        # If the mask is all False, it indicates a potential deadlock or bug.
        if not np.any(valid_mask):
            # Try to force PASS as valid if nothing else is.
            pass_action_id = ACTION_MAP.get(ACTION_PASS)
            if pass_action_id is not None and 0 <= pass_action_id < NUM_POSSIBLE_ACTIONS:
                 print(f"Warning: No valid actions reported by game for {self.current_player.name}. Forcing PASS action ID {pass_action_id} as valid.")
                 valid_mask[pass_action_id] = True
            else:
                 # This is a critical error state
                 print(f"FATAL ERROR: No valid actions found for {self.current_player.name}, and PASS action ID is invalid or missing!")
                 # Depending on training setup, might want to raise error or just return the all-false mask
                 # raise RuntimeError("No valid actions possible, including PASS.")
        return valid_mask

    def step(self, action: int):
        """
        Executes one time step within the environment.
        Returns win information in the info dict upon termination.
        """
        if self.game is None or self.current_player is None:
            # This should ideally not happen if reset() was called, but handle defensively
            print("ERROR: step() called but game or current_player is None. Resetting might have failed.")
            # Return dummy values consistent with a failed step
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            error_info = {"error": "Step called on uninitialized game", "phase": "error", "is_success": 0.0}
            return dummy_obs, -10.0, True, False, error_info # Terminate episode

        terminated = False
        truncated = False
        reward = 0.0
        info = self._get_info() # Get base info first

        action_str = ID_TO_ACTION.get(action)

        if action_str is None:
            print(f"Warning: Received invalid action ID {action}. Taking no action, assigning penalty.")
            observation = self._get_obs()
            reward = -1.0
            info["error"] = f"Invalid action ID {action} received."
            # Don't terminate here, let the agent try again if needed
        else:
            try:
                acting_player_before_step = self.current_player # Store who acted

                if self.render_mode == "human":
                    print(f"\n>>> {acting_player_before_step.name} attempts action: {action_str} (ID: {action})")

                # --- Execute Step ---
                _next_state_dict, step_reward, terminated = self.game.step(action_str)
                reward = float(step_reward)
                self.current_player = self.game.game_state.get_current_player() # Update player ref

                # --- Log Win/Loss Info on Termination ---
                if terminated:
                    winner = self.game.game_state.check_win_condition() # Check winner based on game state
                    # Determine if the player *who took the action* is the winner
                    # This logic might need adjustment based on how check_win_condition is defined
                    # Assuming check_win_condition returns the winning Player object or None
                    if winner is None:
                         # Could be a draw (e.g., turn limit) or error
                         info["is_success"] = 0.0 # Treat draw/error as not success for this agent
                         info["winner"] = "Draw/Error"
                    elif winner == acting_player_before_step:
                        info["is_success"] = 1.0 # The agent who acted won
                        info["winner"] = acting_player_before_step.name
                    else:
                        info["is_success"] = 0.0 # The agent who acted lost
                        info["winner"] = winner.name # Log the actual winner's name

                # --- Get Observation for the NEW state ---
                observation = self._get_obs()
                # Add any other info AFTER getting the obs for the new state
                info.update(self._get_info()) # Update with current turn info etc.


            except Exception as e:
                print(f"FATAL ERROR during game.step with action '{action_str}' (ID: {action})")
                import traceback
                traceback.print_exc()
                observation = np.zeros_like(self.observation_space.sample()) # Return dummy obs
                reward = -10.0
                terminated = True
                info["error"] = f"Exception during game step: {e}"
                info["is_success"] = 0.0 # Failed episode
                # Return immediately

        # --- Rendering ---
        if self.render_mode == "human":
            self.render()
            print(f"<<< Action Result: Reward={reward}, Terminated={terminated}")
        elif self.render_mode == "text":
             pass # Text rendering can be added here or handled by prints within step

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        print(f"[{os.getpid()}] DEBUG: Entering PokemonTCGPocketEnv.reset") # Add this
        super().reset(seed=seed)
        print(f"[{os.getpid()}] DEBUG: Reset: Calling _setup_new_game...")
        try: # Add try/except
            self._setup_new_game()
        except Exception as e:
            print(f"[{os.getpid()}] ERROR IN _setup_new_game: {e}")
            traceback.print_exc()
            # You might want to return something default here or re-raise
            # Returning None might be causing the TypeError directly
            # Let's re-raise to see the original error more clearly
            raise
        print(f"[{os.getpid()}] DEBUG: Reset: _setup_new_game finished. Calling _get_obs...")
        try: # Add try/except
            observation = self._get_obs()
        except Exception as e:
            print(f"[{os.getpid()}] ERROR IN _get_obs: {e}")
            traceback.print_exc()
            raise # Re-raise
        print(f"[{os.getpid()}] DEBUG: Reset: _get_obs finished. Calling _get_info...")
        try: # Add try/except
            info = self._get_info()
        except Exception as e:
            print(f"[{os.getpid()}] ERROR IN _get_info: {e}")
            traceback.print_exc()
            raise # Re-raise
        print(f"[{os.getpid()}] DEBUG: Reset: _get_info finished. Returning observation and info.")
        return observation, info

    def _get_info(self) -> Dict[str, Any]:
         """ Returns auxiliary information (called AFTER state might have changed). """
         return {
              "current_player_name": self.current_player.name if self.current_player else "N/A",
              "turn_number": self.game.game_state.turn_number if self.game else 0,
              # Removed action_mask, SB3-Contrib handles it via action_mask_fn
         }


    def render(self):
        """Renders the environment based on the selected render_mode."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "text":
            # Text rendering might be done incrementally in step, or snapshot here
             pass # Already handled in step for text mode in this example


    def _render_human(self):
        """Provides a detailed text-based rendering for human observation."""
        if self.game is None or self.current_player is None:
             print("Cannot render: Game not initialized.")
             return

        print("\n" + "="*40)
        gs = self.game.game_state
        p1 = self.game.player1
        p2 = self.game.player2
        current_player_render = gs.get_current_player() # Use game's current player

        print(f"Turn: {gs.turn_number} | Current Player: {current_player_render.name}")
        print(f"First Turn? {'Yes' if gs.is_first_turn else 'No'}")
        print("-" * 20)

        # --- Player 1 Info ---
        print(f"Player 1: {p1.name}")
        print(f"  Points: {p1.points}/{POINTS_TO_WIN}")
        print(f"  Hand: {len(p1.hand)} | Deck: {len(p1.deck)} | Discard: {len(p1.discard_pile)}")
        print(f"  Energy Stand: Avail='{p1.energy_stand_available or 'None'}', Preview='{p1.energy_stand_preview or 'None'}'")
        print(f"  Active: {p1.active_pokemon}") # Uses Pokemon.__repr__
        bench_str = ", ".join(repr(p) for p in p1.bench) if p1.bench else "Empty"
        print(f"  Bench ({len(p1.bench)}/{MAX_BENCH_SIZE}): [{bench_str}]")
        # Optionally print hand cards for debugging (can be long)
        # print(f"  Hand Cards: {p1.hand}")
        print("-" * 20)

        # --- Player 2 Info ---
        print(f"Player 2: {p2.name}")
        print(f"  Points: {p2.points}/{POINTS_TO_WIN}")
        print(f"  Hand: {len(p2.hand)} | Deck: {len(p2.deck)} | Discard: {len(p2.discard_pile)}")
        print(f"  Energy Stand: Avail='{p2.energy_stand_available or 'None'}', Preview='{p2.energy_stand_preview or 'None'}'")
        print(f"  Active: {p2.active_pokemon}") # Uses Pokemon.__repr__
        bench_str = ", ".join(repr(p) for p in p2.bench) if p2.bench else "Empty"
        print(f"  Bench ({len(p2.bench)}/{MAX_BENCH_SIZE}): [{bench_str}]")
        # print(f"  Hand Cards: {p2.hand}") # Optionally print hand
        print("-" * 20)

        # --- Possible Actions & Mask Info ---
        print(f"Possible Actions for {current_player_render.name}:")
        possible_actions = self.game.get_possible_actions() # Get fresh list
        for pa in possible_actions:
            print(f"  - {pa}")

        # Display action mask for debugging
        try:
            mask = self.action_mask_fn()
            valid_action_ids = np.where(mask)[0]
            valid_action_names = [ID_TO_ACTION.get(idx, f'INVALID_ID_{idx}') for idx in valid_action_ids]
            print("\nAction Mask Info (for Agent):")
            print(f"  Valid Action IDs: {valid_action_ids.tolist()}") # Convert to list for cleaner print
            print(f"  Valid Action Names: {valid_action_names}")
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
        self.current_player = None


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
