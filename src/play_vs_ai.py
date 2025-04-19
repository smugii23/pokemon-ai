import random
import json
import os
import copy
import traceback # For better error reporting
from typing import List, Optional, Dict, Tuple, Any, Callable
import numpy as np # <-- Import numpy

# Assume your existing simulator code (Player, GameState, PokemonCard, etc.)
# and constants (ACTION_*, MAX_HAND_SIZE, etc.) are defined above or imported.
# Make sure the imports from the original code block are present.
from simulator.entities import Player, GameState, PokemonCard, TrainerCard, Card, Attack, POINTS_TO_WIN
from rl_env import ID_TO_ACTION

# --- Constants (Copied from your original code for completeness) ---
CARD_DATA_FILE = os.path.join(os.path.dirname(__file__), 'cards.json') # Adjust path as needed
MAX_HAND_SIZE = 10
MAX_BENCH_SIZE = 3

ACTION_PASS = "PASS"
ACTION_ATTACK_PREFIX = "ATTACK_"
ACTION_ATTACH_ENERGY_ACTIVE = "ATTACH_ENERGY_ACTIVE"
ACTION_ATTACH_ENERGY_BENCH_PREFIX = "ATTACH_ENERGY_BENCH_"
ACTION_PLAY_BASIC_BENCH_PREFIX = "PLAY_BASIC_BENCH_"
ACTION_USE_ABILITY_ACTIVE = "USE_ABILITY_ACTIVE"
ACTION_USE_ABILITY_BENCH_PREFIX = "USE_ABILITY_BENCH_"
ACTION_PLAY_SUPPORTER_PREFIX = "PLAY_SUPPORTER_"
ACTION_PLAY_ITEM_PREFIX = "PLAY_ITEM_"
ACTION_ATTACH_TOOL_ACTIVE = "ATTACH_TOOL_ACTIVE_"
ACTION_ATTACH_TOOL_BENCH_PREFIX = "ATTACH_TOOL_BENCH_"
ACTION_RETREAT_TO_BENCH_PREFIX = "RETREAT_TO_BENCH_"

# --- Target-Specific Trainer Actions ---
ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX = "PLAY_SUPPORTER_CYRUS_TARGET_"
ACTION_PLAY_ITEM_POTION_TARGET_PREFIX = "PLAY_ITEM_POTION_TARGET_"
ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX = "PLAY_SUPPORTER_PCL_TARGET_"
ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX = "PLAY_SUPPORTER_DAWN_SOURCE_TARGET_"

# Setup Phase Actions
ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX = "setup_choose_active_"
ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX = "setup_choose_bench_"
ACTION_SETUP_CONFIRM_READY = "setup_confirm_ready"

# --- Your Game Class (Slightly trimmed for brevity, assume it works as provided) ---
class Game:
    def __init__(self, player1_deck_names: List[str], player2_deck_names: List[str], player1_energy_types: List[str], player2_energy_types: List[str]):
        self.card_data = self._load_card_data()
        if not self.card_data:
            raise ValueError("Failed to load card data. Cannot initialize game.")
        if len(player1_deck_names) != 20 or len(player2_deck_names) != 20:
            print(f"Warning: Input deck name lists should have 20 cards. P1: {len(player1_deck_names)}, P2: {len(player2_deck_names)}")

        self.player1 = Player("Player 1")
        self.player2 = Player("Player 2")
        self.player1.deck_energy_types = player1_energy_types if player1_energy_types else ["Colorless"]
        self.player2.deck_energy_types = player2_energy_types if player2_energy_types else ["Colorless"]

        deck1_cards = self._build_deck(player1_deck_names)
        deck2_cards = self._build_deck(player2_deck_names)

        self.player1.setup_game(deck1_cards)
        self.player2.setup_game(deck2_cards)

        self.game_state = GameState(self.player1, self.player2)
        self.turn_limit = 50 # Increased limit for human play
        self.actions_this_turn: Dict[str, Any] = {}

        print(f"--- Game Start ---")
        print(f"Player 1 Energy Types: {self.player1.deck_energy_types}")
        print(f"Player 2 Energy Types: {self.player2.deck_energy_types}")
        print(f"Starting Player: {self.game_state.get_current_player().name}")
        self._initialize_energy_stand()
        # Setup phase actions handled within the game loop / step function

    # Assume _load_card_data, _build_deck, _initialize_energy_stand, _start_turn
    # get_state_representation, _get_pokemon_details, get_possible_actions,
    # step, _commit_setup_choices, _execute_ability_effect, _execute_trainer_effect
    # methods exist here exactly as you provided them in the original prompt.
    # (Copying them all would make this response excessively long)
    # ... (Paste your full Game class implementation here) ...
    # --- Make sure all methods from your original 'Game' class are included ---
    def _load_card_data(self) -> Dict[str, Dict]:
        """Loads card definitions from the JSON file."""
        try:
            # Use a placeholder path if the original doesn't exist for testing
            if not os.path.exists(CARD_DATA_FILE):
                 print(f"Warning: Card data file not found at {CARD_DATA_FILE}. Using placeholder path.")
                 # Try a common location or raise an error
                 placeholder_path = 'cards.json' # Or './data/cards.json', etc.
                 if not os.path.exists(placeholder_path):
                     print("Error: Placeholder cards.json also not found. Cannot load card data.")
                     return {}
                 actual_path = placeholder_path
            else:
                 actual_path = CARD_DATA_FILE

            with open(actual_path, 'r') as f:
                all_cards_data = json.load(f)
            # Convert list of cards into a dict keyed by name for easy lookup
            card_dict = {card['name']: card for card in all_cards_data}
            print(f"Successfully loaded data for {len(card_dict)} cards from {actual_path}")
            return card_dict
        except FileNotFoundError:
            print(f"Error: Card data file not found at specified paths.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the card data file.")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred loading card data: {e}")
            traceback.print_exc() # Print traceback for debugging
            return {}

    def _build_deck(self, deck_names: List[str]) -> List[Card]:
        """Builds a list of Card objects from a list of card names using loaded data."""
        deck_cards: List[Card] = []
        if not self.card_data:
             print("Error: Cannot build deck, card data is not loaded.")
             return []
        for name in deck_names:
            card_info = self.card_data.get(name)
            if not card_info:
                print(f"Warning: Card name '{name}' not found in loaded card data. Skipping.")
                continue

            # Use deepcopy to ensure each card in the deck is a unique instance
            card_info_copy = copy.deepcopy(card_info)

            # --- Instantiate Attacks ---
            attacks = []
            if card_info_copy.get("attacks"):
                for attack_data in card_info_copy["attacks"]:
                    attacks.append(Attack(
                        name=attack_data.get("name", "Unknown Attack"),
                        cost=attack_data.get("cost", {}),
                        damage=attack_data.get("damage", 0),
                        effect=attack_data.get("effect_tag") # Pass effect tag if present
                    ))

            # --- Instantiate Card based on type ---
            card_type = card_info_copy.get("card_type")

            if card_type == "Pokemon":
                pokemon = PokemonCard(
                    name=card_info_copy.get("name", "Unknown Pokemon"),
                    hp=card_info_copy.get("hp", 0), # Use .get with default
                    attacks=attacks,
                    pokemon_type=card_info_copy.get("pokemon_type", "Colorless"),
                    weakness_type=card_info_copy.get("weakness_type"),
                    retreat_cost=card_info_copy.get("retreat_cost", 0), # Read retreat cost
                    is_ex=card_info_copy.get("is_ex", False),
                    is_basic=card_info_copy.get("is_basic", True), # Assume basic if not specified? Check parser
                    ability=card_info_copy.get("ability") # Pass ability dict if present
                )
                deck_cards.append(pokemon)
            elif card_type in ["Supporter", "Item", "Tool", "Stadium"]: # Handle Trainer types
                trainer = TrainerCard(
                    name=card_info_copy.get("name", "Unknown Trainer"),
                    trainer_type=card_type,
                    effect_tag=card_info_copy.get("effect_tag")
                )
                deck_cards.append(trainer)
            elif card_type == "Energy":
                # Placeholder for potential future EnergyCard implementation
                # print(f"Warning: Energy card '{name}' found, but EnergyCard class not implemented. Skipping.")
                pass # Allow energy cards in list, but they won't be instantiated as special objects yet
            else:
                # Fallback for unknown or missing card types
                print(f"Warning: Card '{name}' has unknown or missing card_type '{card_type}'. Skipping.")
                pass

        return deck_cards


    def _initialize_energy_stand(self):
        """Sets the initial state of the energy stand for both players."""
        self.player1.energy_stand_preview = random.choice(self.player1.deck_energy_types) if self.player1.deck_energy_types else None
        self.player1.energy_stand_available = None

        self.player2.energy_stand_preview = random.choice(self.player2.deck_energy_types) if self.player2.deck_energy_types else None
        self.player2.energy_stand_available = None

    def _start_turn(self):
        """Handles start-of-turn procedures including energy stand update and drawing a card, and triggers passive abilities."""
        player = self.game_state.get_current_player()
        # Check for deck out BEFORE drawing
        if not player.deck:
            opponent = self.game_state.get_opponent()
            print(f"Game Over! {player.name} decked out! Winner: {opponent.name}")
            self.game_state.winner = opponent # Set winner state
            # Consider how to signal 'done' properly here if _start_turn is called within step
            # For now, rely on the check in the main loop or step function after _start_turn completes
            return # Stop further turn start actions if decked out

        print(f"\n--- Turn {self.game_state.turn_number}: {player.name}'s Turn ---")
        self.actions_this_turn = {
            "energy_attached": False,
            "supporter_played": False,
            "retreat_used": False,
            "red_effect_active": False,
            "leaf_effect_active": False
        }

        if player.energy_stand_available is None:
            player.energy_stand_available = player.energy_stand_preview
            player.energy_stand_preview = None

        if player.deck_energy_types:
            player.energy_stand_preview = random.choice(player.deck_energy_types)
        else:
            player.energy_stand_preview = None

        print(f"{player.name} Energy Stand - Available: {player.energy_stand_available or 'None'}, Preview: {player.energy_stand_preview or 'None'}")

        # Draw card at the start of the turn
        player.draw_cards(1)
        # Re-check for deck out immediately AFTER drawing (if draw_cards handles empty deck gracefully)
        # Although the check before drawing is usually sufficient for TCG win condition.


    def get_state_representation(self, player: Player) -> Dict[str, Any]:
        """
        Gather all relevant information about the current game state from the perspective of the
        given player, so that the AI can play/learn.
        """
        opponent = self.game_state.get_opponent() # Pass player to correctly identify opponent

        my_active_details = self._get_pokemon_details(player.active_pokemon)
        my_bench_details = [self._get_pokemon_details(p) for p in player.bench if p] # Filter out Nones implicitly

        opp_active_details = self._get_pokemon_details(opponent.active_pokemon)
        # Filter opponent's bench to only show count, not details (as per rules)
        opp_bench_count = sum(1 for p in opponent.bench if p) # Count non-None slots

        state_dict = {
            # player info
            "my_hand_size": len(player.hand),
            "my_hand_cards": [c.name for c in player.hand],
            "my_deck_size": len(player.deck),
            "my_discard_size": len(player.discard_pile),
            "my_discard_pile_cards": [c.name for c in player.discard_pile],
            "my_points": player.points,
            "my_energy_stand_available": player.energy_stand_available,
            "my_energy_stand_preview": player.energy_stand_preview,
            "my_active_pokemon": my_active_details,
            "my_bench_pokemon": my_bench_details, # List of dicts for player's own bench
            "my_bench_size": len(player.bench), # Actual current bench size

            # opponent info
            "opp_hand_size": len(opponent.hand),
            "opp_deck_size": len(opponent.deck),
            "opp_discard_size": len(opponent.discard_pile),
            "opp_discard_pile_cards": [c.name for c in opponent.discard_pile],
            "opp_points": opponent.points,
            "opp_energy_stand_status": { # Keep this structure
                "available_exists": opponent.energy_stand_available is not None,
                "preview": opponent.energy_stand_preview
            },
            "opp_active_pokemon": opp_active_details,
            "opp_bench_size": opp_bench_count, # Just the count for opponent

            # info about the game
            "turn": self.game_state.turn_number,
            "is_my_turn": player == self.game_state.get_current_player(),
            "can_attach_energy": not self.actions_this_turn.get("energy_attached", False),
            "can_play_supporter": not self.actions_this_turn.get("supporter_played", False), # Add this
            "can_retreat": not self.actions_this_turn.get("retreat_used", False), # Add this
            "is_first_turn": self.game_state.is_first_turn, # Still useful for rules
            "setup_phase_active": not player.setup_ready or not opponent.setup_ready, # Flag for setup
            "my_setup_ready": player.setup_ready, # Own readiness
            "opponent_setup_ready": opponent.setup_ready, # Opponent readiness
        }
        return state_dict


    def _get_pokemon_details(self, pokemon: Optional[PokemonCard]) -> Dict[str, Any]:
        """
        Extract relevant details from a PokemonCard object for state representation.
        Handles None input.
        """
        if pokemon is None:
            return {} # Return empty dict for None Pokemon

        # Basic details always available
        details = {
            "name": pokemon.name,
            "hp": pokemon.hp,
            "current_hp": pokemon.current_hp,
            "attached_energy": copy.deepcopy(pokemon.attached_energy), # Deep copy for safety
            "is_fainted": pokemon.is_fainted,
            "attached_tool_name": pokemon.attached_tool.name if pokemon.attached_tool else None,
            "is_ex": pokemon.is_ex, # Useful info
            "pokemon_type": pokemon.pokemon_type, # Useful info
            "weakness_type": pokemon.weakness_type, # Useful info
            "retreat_cost": pokemon.retreat_cost, # Useful info
            "ability_name": pokemon.ability.get("name") if pokemon.ability else None, # Ability presence/name
            "ability_type": pokemon.ability.get("type") if pokemon.ability else None,
        }

        # Attacks - conditionally add details if needed by AI, otherwise just names
        details["attack_names"] = [attack.name for attack in pokemon.attacks]
        # Optionally add more attack details like cost/damage if the AI needs it directly
        # details["attacks_details"] = [
        #     {"name": a.name, "cost": a.cost, "damage": a.damage} for a in pokemon.attacks
        # ]

        return details

    def get_possible_actions(self) -> List[str]:
        """
        Get a list of valid actions for the current player.
        (Assumes the implementation from the original prompt is correct)
        """
        player = self.game_state.get_current_player()
        opponent = self.game_state.get_opponent()
        actions = []
        player_idx = self.game_state.current_player_index

        # --- Simultaneous Setup Phase Logic ---
        is_setup_phase_check = not player.setup_ready or not opponent.setup_ready
        if is_setup_phase_check:
            if not player.setup_ready:
                assigned_hand_indices = set()
                pending_active_card_instance = player.pending_setup_active
                pending_bench_card_instances = player.pending_setup_bench

                # Find indices of already assigned cards
                if pending_active_card_instance:
                    try:
                        active_index = player.hand.index(pending_active_card_instance)
                        assigned_hand_indices.add(active_index)
                    except ValueError: pass # Should not happen if logic is correct

                for bench_card_instance in pending_bench_card_instances:
                    try:
                        bench_index = player.hand.index(bench_card_instance)
                        assigned_hand_indices.add(bench_index)
                    except ValueError: pass # Should not happen

                # 1. Must choose Active if not chosen
                if pending_active_card_instance is None:
                    found_basic = False
                    for i, card in enumerate(player.hand):
                        if isinstance(card, PokemonCard) and card.is_basic:
                            action_name = f"{ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX}{i}"
                            actions.append(action_name)
                            found_basic = True
                    if not found_basic:
                         # This indicates a deck building or mulligan issue not handled here
                         print(f"CRITICAL ERROR: Player {player.name} has no basic Pokemon in hand during setup action generation!")
                         return [ACTION_PASS] # Allow passing to potentially trigger mulligan or error state
                    return actions # Only allow choosing active

                # 2. Can choose Bench (if Active chosen and bench not full)
                can_add_to_bench = len(pending_bench_card_instances) < MAX_BENCH_SIZE
                if can_add_to_bench:
                    for i, card in enumerate(player.hand):
                        if i not in assigned_hand_indices and isinstance(card, PokemonCard) and card.is_basic:
                            action_name = f"{ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX}{i}"
                            actions.append(action_name)

                # 3. Can confirm ready if Active has been chosen
                if pending_active_card_instance is not None:
                    actions.append(ACTION_SETUP_CONFIRM_READY)

                # Always allow pass during setup choice phase? Or only if ready?
                # Allowing pass here might let player skip benching optionally.
                # Let's require CONFIRM_READY instead of PASS to finish setup.
                # actions.append(ACTION_PASS) # Maybe remove this?

                # If only action is CONFIRM_READY, return that. Otherwise return choices.
                if not actions: # If no bench options left, only confirm should be possible
                     if pending_active_card_instance is not None: actions.append(ACTION_SETUP_CONFIRM_READY)
                     else: actions.append(ACTION_PASS) # Should not happen if active must be chosen

                return actions if actions else [ACTION_PASS] # Fallback pass

            else: # Player is ready, opponent is not
                return [ACTION_PASS] # Must wait

        # --- Normal Turn Actions ---
        can_attach_energy_this_turn = not self.actions_this_turn.get("energy_attached", False)
        is_game_first_turn = self.game_state.is_first_turn
        can_play_supporter = not self.actions_this_turn.get("supporter_played", False)
        can_retreat_this_turn = not self.actions_this_turn.get("retreat_used", False)

        # Attach Energy
        if player.energy_stand_available and can_attach_energy_this_turn and not is_game_first_turn:
            if player.active_pokemon and not player.active_pokemon.is_fainted:
                actions.append(ACTION_ATTACH_ENERGY_ACTIVE)
            for i, bench_pokemon in enumerate(player.bench):
                if bench_pokemon and not bench_pokemon.is_fainted:
                    actions.append(f"{ACTION_ATTACH_ENERGY_BENCH_PREFIX}{i}")

        # Attack
        if player.active_pokemon and not player.active_pokemon.is_fainted and not is_game_first_turn:
            for i, attack in enumerate(player.active_pokemon.attacks):
                if player.active_pokemon.can_attack(i):
                    actions.append(f"{ACTION_ATTACK_PREFIX}{i}")

        # Play Basic to Bench
        if player.can_place_on_bench():
            for i, card in enumerate(player.hand):
                if isinstance(card, PokemonCard) and card.is_basic:
                    actions.append(f"{ACTION_PLAY_BASIC_BENCH_PREFIX}{i}")

        # Use Active Ability
        if player.active_pokemon and player.active_pokemon.ability:
            ability_data = player.active_pokemon.ability
            ability_name = ability_data.get("name")
            ability_type = ability_data.get("type")
            ability_used_key = f"ability_used_{ability_name}" # Simple key, assumes unique names for now
            if ability_type == "Active" and not self.actions_this_turn.get(ability_used_key, False):
                # Add specific checks if ability has conditions (e.g., requires discard, specific state)
                actions.append(ACTION_USE_ABILITY_ACTIVE)

        # Use Bench Ability
        for i, bench_pokemon in enumerate(player.bench):
            if bench_pokemon and bench_pokemon.ability:
                ability_data = bench_pokemon.ability
                ability_name = ability_data.get("name")
                ability_type = ability_data.get("type")
                ability_used_key = f"ability_used_{ability_name}"
                if ability_type == "Active" and not self.actions_this_turn.get(ability_used_key, False):
                    # Add specific checks if ability has conditions
                    actions.append(f"{ACTION_USE_ABILITY_BENCH_PREFIX}{i}")

        # Play Trainers
        for i, card in enumerate(player.hand):
            if isinstance(card, TrainerCard):
                effect_tag = card.effect_tag
                if card.trainer_type == "Supporter":
                    if can_play_supporter:
                        # --- Specific Supporter Condition Checks ---
                        can_play_this = True
                        if effect_tag == "TRAINER_SUPPORTER_SABRINA_SWITCH_OUT_YOUR_OPPONENT":
                             has_valid_bench = any(p and not p.is_fainted for p in opponent.bench)
                             if not opponent.active_pokemon or not has_valid_bench: can_play_this = False
                        elif effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
                             has_damaged_bench = any(p and not p.is_fainted and p.current_hp < p.hp for p in opponent.bench)
                             if not opponent.active_pokemon or not has_damaged_bench: can_play_this = False
                        elif effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
                             has_benched_energy = any(p and not p.is_fainted and p.attached_energy for p in player.bench)
                             if not player.active_pokemon or player.active_pokemon.is_fainted or not has_benched_energy: can_play_this = False
                        elif effect_tag == "TRAINER_SUPPORTER_TEAM_ROCKET_GRUNT_FLIP_A_COIN_UN":
                             if not opponent.active_pokemon or opponent.active_pokemon.is_fainted or not opponent.active_pokemon.attached_energy: can_play_this = False
                        elif effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
                             is_any_pokemon_damaged = (player.active_pokemon and player.active_pokemon.current_hp < player.active_pokemon.hp) or \
                                                     any(p and p.current_hp < p.hp for p in player.bench)
                             if not is_any_pokemon_damaged: can_play_this = False
                        # Add other supporter conditions here...

                        # --- Generate Target-Specific or Generic Actions ---
                        if can_play_this:
                            if effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
                                for bench_idx, p in enumerate(opponent.bench):
                                    if p and not p.is_fainted and p.current_hp < p.hp:
                                        actions.append(f"{ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX}{i}_{bench_idx}")
                            elif effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
                                if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.current_hp < player.active_pokemon.hp:
                                    actions.append(f"{ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX}{i}_active")
                                for bench_idx, p in enumerate(player.bench):
                                    if p and not p.is_fainted and p.current_hp < p.hp:
                                        actions.append(f"{ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX}{i}_bench_{bench_idx}")
                            elif effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
                                if player.active_pokemon and not player.active_pokemon.is_fainted:
                                    for bench_idx, p in enumerate(player.bench):
                                        if p and not p.is_fainted and p.attached_energy:
                                            actions.append(f"{ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX}{i}_{bench_idx}")
                            else: # Generic supporter
                                actions.append(f"{ACTION_PLAY_SUPPORTER_PREFIX}{i}")

                elif card.trainer_type == "Item":
                    # --- Specific Item Condition Checks ---
                    can_play_this = True
                    if effect_tag == "TRAINER_ITEM_POTION_HEAL_20_DAMAGE_FROM_1_OF_YOUR_":
                        is_any_pokemon_damaged = (player.active_pokemon and player.active_pokemon.current_hp < player.active_pokemon.hp) or \
                                                 any(p and p.current_hp < p.hp for p in player.bench)
                        if not is_any_pokemon_damaged: can_play_this = False
                    elif effect_tag == "TRAINER_ITEM_POKÉ_BALL_PUT_1_RANDOM_BASIC_POKEMON_":
                        # Check if deck is empty - technically playable but fails
                        if not player.deck: can_play_this = False # Or allow and let effect handle it? Let's allow.
                        pass # No pre-condition other than having a deck

                    # Add other item conditions here...

                    # --- Generate Target-Specific or Generic Actions ---
                    if can_play_this:
                        if effect_tag == "TRAINER_ITEM_POTION_HEAL_20_DAMAGE_FROM_1_OF_YOUR_":
                            if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.current_hp < player.active_pokemon.hp:
                                actions.append(f"{ACTION_PLAY_ITEM_POTION_TARGET_PREFIX}{i}_active")
                            for bench_idx, p in enumerate(player.bench):
                                if p and not p.is_fainted and p.current_hp < p.hp:
                                    actions.append(f"{ACTION_PLAY_ITEM_POTION_TARGET_PREFIX}{i}_bench_{bench_idx}")
                        else: # Generic item
                            actions.append(f"{ACTION_PLAY_ITEM_PREFIX}{i}")

                elif card.trainer_type == "Tool":
                    # Check potential targets
                    if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.attached_tool is None:
                        actions.append(f"{ACTION_ATTACH_TOOL_ACTIVE}{i}")
                    for bench_idx, bench_pokemon in enumerate(player.bench):
                        if bench_pokemon and not bench_pokemon.is_fainted and bench_pokemon.attached_tool is None:
                            actions.append(f"{ACTION_ATTACH_TOOL_BENCH_PREFIX}{bench_idx}_{i}")

                # Add Stadium logic if implemented

        # Retreat
        if can_retreat_this_turn and player.active_pokemon and not player.active_pokemon.is_fainted:
            valid_bench_targets = [i for i, p in enumerate(player.bench) if p and not p.is_fainted]
            if valid_bench_targets:
                retreat_cost_modifier = -2 if self.actions_this_turn.get("leaf_effect_active", False) else 0
                if player.active_pokemon.can_retreat(current_turn_cost_modifier=retreat_cost_modifier):
                    for bench_idx in valid_bench_targets:
                        actions.append(f"{ACTION_RETREAT_TO_BENCH_PREFIX}{bench_idx}")

        # Pass
        actions.append(ACTION_PASS)

        # Filter out duplicates just in case
        actions = sorted(list(set(actions)))

        return actions


    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool]:
        """
        Execute an action, update the game state, and return the new state, reward, and done flag.
        State returned is for the player whose turn it is *next*.
        (Assumes the implementation from the original prompt is correct, including reward logic)
        """
        player = self.game_state.get_current_player()
        opponent = self.game_state.get_opponent()
        initial_player_points = player.points
        initial_opponent_points = opponent.points
        reward = 0.0 # Base reward per step
        done = False
        action_executed = False
        action_was_known = False # Track if the action string format was recognized
        is_turn_ending_action = False # Track if action inherently ends turn
        original_action = action # Keep track for ability turn-end logic

        # Check for immediate win/loss conditions not tied to action (e.g., deck out at turn start)
        if self.game_state.winner:
            done = True
            winner = self.game_state.winner
            if winner == player: reward = 2.0
            elif winner == opponent: reward = -2.0
            else: reward = -1.0 # Draw or error?
            next_state = self.get_state_representation(player) # Return state for the player who would have acted
            return next_state, reward, done

        is_setup_phase = not player.setup_ready or not opponent.setup_ready

        try: # Wrap action execution in try-except for better debugging
            # --- Handle Setup Phase Actions ---
            if is_setup_phase:
                action_was_known = True # Assume setup actions are known patterns
                if not player.setup_ready: # Actions for player not ready
                    if action.startswith(ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX):
                        if player.pending_setup_active is not None:
                             print(f"Invalid Action: Active already chosen.")
                             reward -= 0.1
                        else:
                            try:
                                hand_index = int(action.split('_')[-1])
                                card_to_choose = player.hand[hand_index]
                                if isinstance(card_to_choose, PokemonCard) and card_to_choose.is_basic and card_to_choose not in player.pending_setup_bench:
                                    player.pending_setup_active = card_to_choose
                                    print(f"{player.name} chose {card_to_choose.name} as pending Active.")
                                    action_executed = True
                                    # reward += 0.01 # Small reward for valid setup step
                                else:
                                    print(f"Invalid Action: Card {hand_index} is not a valid Basic Pokemon choice.")
                                    reward -= 0.1
                            except (IndexError, ValueError):
                                print(f"Invalid Action: Malformed setup action {action}")
                                reward -= 0.1
                    elif action.startswith(ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX):
                        if player.pending_setup_active is None:
                            print(f"Invalid Action: Choose Active first.")
                            reward -= 0.1
                        elif len(player.pending_setup_bench) >= MAX_BENCH_SIZE:
                            print(f"Invalid Action: Bench is full.")
                            reward -= 0.1
                        else:
                            try:
                                hand_index = int(action.split('_')[-1])
                                card_to_choose = player.hand[hand_index]
                                already_chosen = (card_to_choose == player.pending_setup_active or card_to_choose in player.pending_setup_bench)
                                if isinstance(card_to_choose, PokemonCard) and card_to_choose.is_basic and not already_chosen:
                                    player.pending_setup_bench.append(card_to_choose)
                                    print(f"{player.name} added {card_to_choose.name} to pending Bench.")
                                    action_executed = True
                                    # reward += 0.01
                                else:
                                    print(f"Invalid Action: Card {hand_index} is not a valid Basic Pokemon choice or already chosen.")
                                    reward -= 0.1
                            except (IndexError, ValueError):
                                print(f"Invalid Action: Malformed setup action {action}")
                                reward -= 0.1
                    elif action == ACTION_SETUP_CONFIRM_READY:
                        if player.pending_setup_active is None:
                            print(f"Invalid Action: Must choose Active before confirming.")
                            reward -= 0.1
                        else:
                            player.setup_ready = True
                            action_executed = True
                            print(f"{player.name} confirmed setup readiness.")
                            if opponent.setup_ready:
                                print("Both players ready! Committing setup choices.")
                                self._commit_setup_choices()
                                print("Setup complete. Starting first turn.")
                                # _start_turn is called by switch_turn logic below
                                # Explicitly switch turn here? No, let normal logic handle it.
                            else:
                                # Don't switch turn automatically here, let the main logic handle player switching
                                pass
                    elif action == ACTION_PASS:
                         print(f"Invalid Action: Cannot pass during your setup choice phase. Choose actions or confirm ready.")
                         reward -= 0.1
                    else: # Unknown action during setup choice
                        action_was_known = False
                        print(f"Invalid Action: Unknown action '{action}' during setup.")
                        reward -= 0.1

                else: # Player is ready, opponent is not
                    if action == ACTION_PASS:
                        print(f"{player.name} is ready and passes, waiting for {opponent.name}.")
                        action_executed = True
                        # The turn should switch to the opponent. This happens naturally below.
                    else:
                        print(f"Invalid Action: Player is ready, opponent is not. Only PASS is allowed.")
                        reward -= 0.1

            # --- Handle Normal Turn Actions ---
            elif not is_setup_phase:
                action_was_known = True # Assume known format unless it matches no prefix
                is_game_first_turn_overall = self.game_state.is_first_turn

                if action == ACTION_PASS:
                    print("Turn passed.")
                    action_executed = True
                    is_turn_ending_action = True

                elif action == ACTION_ATTACH_ENERGY_ACTIVE:
                    if not player.active_pokemon: reward = -0.1; print("Invalid: No active Pokemon.")
                    elif is_game_first_turn_overall: reward = -0.1; print("Invalid: Cannot attach energy on first turn.")
                    elif self.actions_this_turn.get("energy_attached"): reward = -0.1; print("Invalid: Energy already attached.")
                    elif not player.energy_stand_available: reward = -0.1; print("Invalid: No energy available.")
                    else:
                        energy_to_attach = player.energy_stand_available
                        player.active_pokemon.attach_energy(energy_to_attach)
                        player.energy_stand_available = None
                        self.actions_this_turn["energy_attached"] = True
                        action_executed = True
                        reward += 0.02
                        print(f"Attached {energy_to_attach} energy to Active {player.active_pokemon.name}.")
                        # Check for Nightmare Aura
                        if player.active_pokemon.name == "Darkrai ex" and energy_to_attach == "Darkness":
                             print(f"Ability Trigger: {player.active_pokemon.name}'s Nightmare Aura!")
                             if opponent.active_pokemon and not opponent.active_pokemon.is_fainted:
                                 opponent.active_pokemon.take_damage(20)
                                 print(f"  Nightmare Aura dealt 20 damage to {opponent.active_pokemon.name}.")
                                 if opponent.active_pokemon.is_fainted:
                                     points_scored = 2 if opponent.active_pokemon.is_ex else 1
                                     player.add_point(points_scored)
                                     print(f"  {opponent.active_pokemon.name} fainted! Scored {points_scored} points.")
                                     # Check win/promote handled after action

                elif action.startswith(ACTION_ATTACH_ENERGY_BENCH_PREFIX):
                    if is_game_first_turn_overall: reward = -0.1; print("Invalid: Cannot attach energy on first turn.")
                    elif self.actions_this_turn.get("energy_attached"): reward = -0.1; print("Invalid: Energy already attached.")
                    elif not player.energy_stand_available: reward = -0.1; print("Invalid: No energy available.")
                    else:
                         try:
                             bench_index = int(action.split('_')[-1])
                             target_pokemon = player.bench[bench_index]
                             if target_pokemon and not target_pokemon.is_fainted:
                                 energy_to_attach = player.energy_stand_available
                                 target_pokemon.attach_energy(energy_to_attach)
                                 player.energy_stand_available = None
                                 self.actions_this_turn["energy_attached"] = True
                                 action_executed = True
                                 reward += 0.02
                                 print(f"Attached {energy_to_attach} energy to Bench {bench_index} ({target_pokemon.name}).")
                                 # Check for Nightmare Aura
                                 if target_pokemon.name == "Darkrai ex" and energy_to_attach == "Darkness":
                                      print(f"Ability Trigger: {target_pokemon.name}'s Nightmare Aura!")
                                      if opponent.active_pokemon and not opponent.active_pokemon.is_fainted:
                                          opponent.active_pokemon.take_damage(20)
                                          print(f"  Nightmare Aura dealt 20 damage to {opponent.active_pokemon.name}.")
                                          if opponent.active_pokemon.is_fainted:
                                              points_scored = 2 if opponent.active_pokemon.is_ex else 1
                                              player.add_point(points_scored)
                                              print(f"  {opponent.active_pokemon.name} fainted! Scored {points_scored} points.")
                                              # Check win/promote handled after action
                             else: reward = -0.1; print(f"Invalid: Bench {bench_index} empty or fainted.")
                         except (IndexError, ValueError): reward = -0.1; print(f"Invalid action format: {action}")

                elif action.startswith(ACTION_ATTACK_PREFIX):
                    if is_game_first_turn_overall: reward = -0.1; print("Invalid: Cannot attack on first turn.")
                    elif not player.active_pokemon or player.active_pokemon.is_fainted: reward = -0.1; print("Invalid: No valid active Pokemon.")
                    else:
                        try:
                            attack_index = int(action.split('_')[-1])
                            if player.active_pokemon.can_attack(attack_index):
                                attack = player.active_pokemon.attacks[attack_index]
                                print(f"{player.name}'s {player.active_pokemon.name} uses {attack.name}!")
                                damage_modifier = 20 if self.actions_this_turn.get("red_effect_active", False) else 0
                                damage_dealt = player.active_pokemon.perform_attack(attack_index, opponent.active_pokemon, damage_modifier)
                                reward += damage_dealt * 0.005 # Smaller reward per damage point
                                action_executed = True
                                is_turn_ending_action = True
                                # Check KO/win handled after action
                            else: reward = -0.1; print(f"Invalid: Cannot afford or use attack {attack_index}.")
                        except (IndexError, ValueError): reward = -0.1; print(f"Invalid action format: {action}")

                elif action.startswith(ACTION_PLAY_BASIC_BENCH_PREFIX):
                    if not player.can_place_on_bench(): reward = -0.1; print("Invalid: Bench is full.")
                    else:
                         try:
                             hand_index = int(action.split('_')[-1])
                             card_to_play = player.hand[hand_index]
                             if isinstance(card_to_play, PokemonCard) and card_to_play.is_basic:
                                 played_card = player.hand.pop(hand_index)
                                 player.place_on_bench(played_card)
                                 print(f"Played {played_card.name} to the bench.")
                                 action_executed = True
                                 reward += 0.01
                             else: reward = -0.1; print(f"Invalid: Card {hand_index} not a Basic Pokemon.")
                         except (IndexError, ValueError): reward = -0.1; print(f"Invalid action format: {action}")

                elif action == ACTION_USE_ABILITY_ACTIVE:
                    source_pokemon = player.active_pokemon
                    if not source_pokemon or not source_pokemon.ability: reward = -0.1; print("Invalid: Active has no ability.")
                    else:
                        ability_data = source_pokemon.ability
                        ability_name = ability_data.get("name", "Unknown Ability")
                        ability_type = ability_data.get("type")
                        ability_used_key = f"ability_used_{ability_name}"
                        if ability_type != "Active": reward = -0.1; print(f"Invalid: Ability '{ability_name}' is not Active type.")
                        elif self.actions_this_turn.get(ability_used_key, False): reward = -0.1; print(f"Invalid: Ability '{ability_name}' already used.")
                        else:
                            print(f"{player.name} uses ability from Active: {ability_name}")
                            effect_tag = ability_data.get("effect_tag")
                            effect_executed, ends_turn = self._execute_ability_effect(effect_tag, player, opponent, source_pokemon)
                            if effect_executed:
                                self.actions_this_turn[ability_used_key] = True
                                action_executed = True
                                reward += 0.03
                                if ends_turn:
                                     print(f"Ability {ability_name} ends the turn.")
                                     is_turn_ending_action = True
                                     action = ACTION_PASS # Ensure turn switch logic triggers
                            else:
                                reward -= 0.1 # Penalize trying unusable ability effect
                                print(f"Ability '{ability_name}' effect could not be executed.")

                elif action.startswith(ACTION_USE_ABILITY_BENCH_PREFIX):
                     try:
                         bench_index = int(action.split('_')[-1])
                         source_pokemon = player.bench[bench_index]
                         if not source_pokemon or not source_pokemon.ability: reward = -0.1; print(f"Invalid: Bench {bench_index} has no ability.")
                         else:
                             ability_data = source_pokemon.ability
                             ability_name = ability_data.get("name", "Unknown Ability")
                             ability_type = ability_data.get("type")
                             ability_used_key = f"ability_used_{ability_name}"
                             if ability_type != "Active": reward = -0.1; print(f"Invalid: Ability '{ability_name}' is not Active type.")
                             elif self.actions_this_turn.get(ability_used_key, False): reward = -0.1; print(f"Invalid: Ability '{ability_name}' already used.")
                             else:
                                 print(f"{player.name} uses ability from Bench {bench_index}: {ability_name}")
                                 effect_tag = ability_data.get("effect_tag")
                                 effect_executed, ends_turn = self._execute_ability_effect(effect_tag, player, opponent, source_pokemon)
                                 if effect_executed:
                                     self.actions_this_turn[ability_used_key] = True
                                     action_executed = True
                                     reward += 0.03
                                     if ends_turn:
                                         print(f"Ability {ability_name} ends the turn.")
                                         is_turn_ending_action = True
                                         action = ACTION_PASS # Ensure turn switch logic triggers
                                 else:
                                     reward -= 0.1 # Penalize trying unusable ability effect
                                     print(f"Ability '{ability_name}' effect could not be executed.")
                     except (IndexError, ValueError): reward = -0.1; print(f"Invalid action format: {action}")

                # --- Play Trainer Cards (Consolidated Logic) ---
                elif action.startswith((ACTION_PLAY_SUPPORTER_PREFIX, ACTION_PLAY_ITEM_PREFIX, ACTION_ATTACH_TOOL_ACTIVE, ACTION_ATTACH_TOOL_BENCH_PREFIX)):
                    target_id = None
                    target_bench_index = None
                    source_bench_index = None
                    card_type_prefix = ""

                    # Determine card type and parse indices
                    try:
                        if action.startswith(ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX):
                            card_type_prefix = ACTION_PLAY_SUPPORTER_PREFIX
                            parts = action.replace(ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX, "").split('_')
                            hand_index = int(parts[0])
                            target_bench_index = int(parts[1])
                        elif action.startswith(ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX):
                            card_type_prefix = ACTION_PLAY_SUPPORTER_PREFIX
                            parts = action.replace(ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX, "").split('_')
                            hand_index = int(parts[0])
                            target_id = "_".join(parts[1:])
                        elif action.startswith(ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX):
                             card_type_prefix = ACTION_PLAY_SUPPORTER_PREFIX
                             parts = action.replace(ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX, "").split('_')
                             hand_index = int(parts[0])
                             source_bench_index = int(parts[1])
                        elif action.startswith(ACTION_PLAY_ITEM_POTION_TARGET_PREFIX):
                            card_type_prefix = ACTION_PLAY_ITEM_PREFIX
                            parts = action.replace(ACTION_PLAY_ITEM_POTION_TARGET_PREFIX, "").split('_')
                            hand_index = int(parts[0])
                            target_id = "_".join(parts[1:])
                        elif action.startswith(ACTION_PLAY_SUPPORTER_PREFIX):
                             card_type_prefix = ACTION_PLAY_SUPPORTER_PREFIX
                             hand_index = int(action.split('_')[-1])
                        elif action.startswith(ACTION_PLAY_ITEM_PREFIX):
                             card_type_prefix = ACTION_PLAY_ITEM_PREFIX
                             hand_index = int(action.split('_')[-1])
                        elif action.startswith(ACTION_ATTACH_TOOL_ACTIVE):
                             card_type_prefix = ACTION_ATTACH_TOOL_ACTIVE # Use specific prefix temporarily
                             hand_index = int(action.split('_')[-1])
                        elif action.startswith(ACTION_ATTACH_TOOL_BENCH_PREFIX):
                             card_type_prefix = ACTION_ATTACH_TOOL_BENCH_PREFIX # Use specific prefix temporarily
                             parts = action.split('_')
                             target_bench_index = int(parts[-2]) # Bench index for tool target
                             hand_index = int(parts[-1])
                        else:
                             raise ValueError("Unknown trainer action prefix")

                        # Validate hand index
                        if not (0 <= hand_index < len(player.hand)):
                             raise IndexError("Hand index out of bounds")

                        card_to_play = player.hand[hand_index]

                        # --- Tool Attachment Logic ---
                        if card_type_prefix == ACTION_ATTACH_TOOL_ACTIVE:
                            if not isinstance(card_to_play, TrainerCard) or card_to_play.trainer_type != "Tool": reward = -0.1; print(f"Invalid: Card {hand_index} is not a Tool.")
                            elif not player.active_pokemon or player.active_pokemon.is_fainted: reward = -0.1; print("Invalid: No valid active target for tool.")
                            elif player.active_pokemon.attached_tool is not None: reward = -0.1; print("Invalid: Active already has a tool.")
                            else:
                                tool_card = player.hand.pop(hand_index)
                                player.active_pokemon.attached_tool = tool_card
                                print(f"Attached {tool_card.name} to active {player.active_pokemon.name}.")
                                # Apply immediate effects (like Giant Cape)
                                if tool_card.effect_tag == "TRAINER_TOOL_GIANT_CAPE_THE_POK_MON_THIS_CARD_IS_ATTACHED_TO_G": # Check tag
                                    player.active_pokemon.hp += 20
                                    player.active_pokemon.current_hp += 20
                                    print(f"  {player.active_pokemon.name} HP increased by 20.")
                                action_executed = True
                                reward += 0.04

                        elif card_type_prefix == ACTION_ATTACH_TOOL_BENCH_PREFIX:
                            if not isinstance(card_to_play, TrainerCard) or card_to_play.trainer_type != "Tool": reward = -0.1; print(f"Invalid: Card {hand_index} is not a Tool.")
                            elif target_bench_index is None or not (0 <= target_bench_index < len(player.bench)): reward = -0.1; print("Invalid: Bad bench index for tool.")
                            else:
                                target_pokemon = player.bench[target_bench_index]
                                if not target_pokemon or target_pokemon.is_fainted: reward = -0.1; print(f"Invalid: Bench {target_bench_index} empty or fainted.")
                                elif target_pokemon.attached_tool is not None: reward = -0.1; print(f"Invalid: Bench {target_bench_index} already has a tool.")
                                else:
                                     tool_card = player.hand.pop(hand_index)
                                     target_pokemon.attached_tool = tool_card
                                     print(f"Attached {tool_card.name} to benched {target_pokemon.name} (Index {target_bench_index}).")
                                     # Apply immediate effects
                                     if tool_card.effect_tag == "TRAINER_TOOL_GIANT_CAPE_THE_POK_MON_THIS_CARD_IS_ATTACHED_TO_G":
                                         target_pokemon.hp += 20
                                         target_pokemon.current_hp += 20
                                         print(f"  {target_pokemon.name} HP increased by 20.")
                                     action_executed = True
                                     reward += 0.04

                        # --- Supporter/Item Play Logic ---
                        elif card_type_prefix in [ACTION_PLAY_SUPPORTER_PREFIX, ACTION_PLAY_ITEM_PREFIX]:
                             expected_type = "Supporter" if card_type_prefix == ACTION_PLAY_SUPPORTER_PREFIX else "Item"
                             if not isinstance(card_to_play, TrainerCard) or card_to_play.trainer_type != expected_type:
                                 reward = -0.1; print(f"Invalid: Card {hand_index} not a {expected_type}.")
                             elif expected_type == "Supporter" and self.actions_this_turn.get("supporter_played", False):
                                 reward = -0.1; print(f"Invalid: Already played a Supporter this turn.")
                             # Add first turn supporter check if needed by rules (usually allowed)
                             # elif expected_type == "Supporter" and is_game_first_turn_overall:
                             #    reward = -0.1; print("Invalid: Cannot play Supporter on first turn.") # Uncomment if rule applies
                             else:
                                 print(f"{player.name} plays {expected_type}: {card_to_play.name}")
                                 # Mark action as executed *before* effect attempt
                                 action_executed = True
                                 if expected_type == "Supporter":
                                     self.actions_this_turn["supporter_played"] = True

                                 # Execute effect
                                 effect_executed = self._execute_trainer_effect(
                                     card_to_play.effect_tag, player, opponent,
                                     target_bench_index=target_bench_index,
                                     target_id=target_id,
                                     source_bench_index=source_bench_index
                                 )

                                 # Remove from hand and discard AFTER attempting effect
                                 played_card = player.hand.pop(hand_index)
                                 player.discard_pile.append(played_card) # Discard even if effect fails

                                 if effect_executed:
                                     reward += 0.05 # Reward for successful trainer effect
                                 else:
                                     # Don't penalize here, penalty is for invalid action choice,
                                     # not for effect failing (e.g., no target found by effect).
                                     print(f"Effect of {card_to_play.name} failed or had no valid target/condition.")


                    except (IndexError, ValueError) as e:
                         reward = -0.1; print(f"Invalid action format or index: {action} ({e})")

                elif action.startswith(ACTION_RETREAT_TO_BENCH_PREFIX):
                    if self.actions_this_turn.get("retreat_used"): reward = -0.1; print("Invalid: Already retreated.")
                    elif not player.active_pokemon or player.active_pokemon.is_fainted: reward = -0.1; print("Invalid: No active to retreat.")
                    else:
                        try:
                            bench_index = int(action.split('_')[-1])
                            if not (0 <= bench_index < len(player.bench)): raise IndexError("Bench index out of range")
                            target_bench_pokemon = player.bench[bench_index]
                            if not target_bench_pokemon or target_bench_pokemon.is_fainted: reward = -0.1; print(f"Invalid: Bench {bench_index} empty or fainted.")
                            else:
                                retreat_cost_modifier = -2 if self.actions_this_turn.get("leaf_effect_active", False) else 0
                                if player.active_pokemon.can_retreat(current_turn_cost_modifier=retreat_cost_modifier):
                                    effective_cost = max(0, player.active_pokemon.retreat_cost + retreat_cost_modifier)
                                    print(f"{player.name} retreats {player.active_pokemon.name} (Cost: {effective_cost}), promotes {target_bench_pokemon.name}.")

                                    player.active_pokemon.discard_energy(effective_cost)

                                    # Swap
                                    current_active_pokemon = player.active_pokemon
                                    player.active_pokemon = target_bench_pokemon
                                    player.bench[bench_index] = current_active_pokemon
                                    player.active_pokemon.is_active = True
                                    current_active_pokemon.is_active = False

                                    self.actions_this_turn["retreat_used"] = True
                                    action_executed = True
                                    reward += 0.01
                                else: reward = -0.1; print(f"Invalid: Cannot afford retreat cost.")
                        except (IndexError, ValueError): reward = -0.1; print(f"Invalid action format: {action}")

                else: # Unknown action format
                    action_was_known = False
                    print(f"Unknown action attempted: {original_action}")
                    reward = -0.1 # Penalize unknown actions more


        except Exception as e:
            print(f"\n!!! ERROR during action execution: {original_action} !!!")
            print(f"Error message: {e}")
            print("Traceback:")
            traceback.print_exc()
            reward = -0.5 # Severe penalty for causing an exception
            action_executed = False # Ensure turn doesn't accidentally proceed normally
            # Potentially end the game here if it's unrecoverable? For now, just penalize.


        # --- Post-Action Checks (KO, Win Conditions, Promote) ---
        # These checks run regardless of action success, as effects (like Nightmare Aura) might cause KOs
        if opponent.active_pokemon and opponent.active_pokemon.is_fainted:
            # Opponent active fainted - handled points inside attack/ability, check win/promote
            print(f"Opponent's active {opponent.active_pokemon.name} fainted.")
            # Discard fainted Pokemon and tool
            fainted_poke = opponent.active_pokemon
            opponent.discard_pile.append(fainted_poke)
            if fainted_poke.attached_tool:
                opponent.discard_pile.append(fainted_poke.attached_tool)
                fainted_poke.attached_tool = None # Clear tool
            opponent.active_pokemon = None # Clear active slot

            winner = self.game_state.check_win_condition()
            if winner:
                print(f"Game Over! Winner: {winner.name} (scored {player.points} points)")
                reward += 2.0 # Big reward for winning KO
                done = True
            elif not done: # If game not over, opponent must promote
                if not opponent.promote_bench_pokemon(): # Handles choosing/moving
                    print(f"Game Over! {opponent.name} has no Pokemon to promote. Winner: {player.name}")
                    reward += 2.0
                    done = True
                else:
                    print(f"{opponent.name} promoted {opponent.active_pokemon.name} to Active.")

        if player.active_pokemon and player.active_pokemon.is_fainted:
            # Player's active fainted (e.g., self-damage, opponent's turn effect?) - less common mid-player-turn
            print(f"Player's active {player.active_pokemon.name} fainted.")
             # Discard fainted Pokemon and tool
            fainted_poke = player.active_pokemon
            player.discard_pile.append(fainted_poke)
            if fainted_poke.attached_tool:
                player.discard_pile.append(fainted_poke.attached_tool)
                fainted_poke.attached_tool = None # Clear tool
            player.active_pokemon = None # Clear active slot

            winner = self.game_state.check_win_condition() # Check if opponent won
            if winner:
                print(f"Game Over! Winner: {winner.name} (opponent scored {opponent.points} points)")
                reward -= 2.0 # Big penalty for losing KO
                done = True
            elif not done: # If game not over, player must promote (though turn likely ends)
                if not player.promote_bench_pokemon():
                    print(f"Game Over! {player.name} has no Pokemon to promote. Winner: {opponent.name}")
                    reward -= 2.0
                    done = True
                else:
                    print(f"{player.name} promoted {player.active_pokemon.name} to Active.")
                    # Turn should still end after this


        # Check win condition again after points/promotion
        if not done:
            winner = self.game_state.check_win_condition()
            if winner:
                print(f"Game Over! Winner: {winner.name} (Points P1:{self.player1.points}, P2:{self.player2.points})")
                if winner == player: reward += 2.0
                else: reward -= 2.0
                done = True

        # Check turn limit
        if not done and self.game_state.turn_number > self.turn_limit:
             print(f"Game Over! Turn limit ({self.turn_limit}) reached.")
             done = True
             # Determine winner by points at turn limit, or declare draw? TCG rules vary.
             if self.player1.points > self.player2.points: self.game_state.winner = self.player1
             elif self.player2.points > self.player1.points: self.game_state.winner = self.player2
             else: print("Draw game by turn limit!") # Handle draw case

             if self.game_state.winner == player: reward += 1.0 # Smaller reward for turn limit win
             elif self.game_state.winner == opponent: reward -= 1.0
             else: reward -= 0.5 # Penalty for draw/limit reached


        # --- Determine if Turn Switches ---
        switch_actual_turn = False
        if not is_setup_phase:
            # Turn ends on PASS, ATTACK, or specific Abilities, or if an unknown/invalid action was forced
            if action_executed and is_turn_ending_action:
                switch_actual_turn = True
            elif not action_executed and not action_was_known: # Penalized unknown action forces turn end
                 print("Ending turn due to unknown action attempt.")
                 switch_actual_turn = True
            # If action_executed is False but action_was_known is True, it was an invalid *known* action
            # (e.g., attach 2nd energy). Player should get another chance, turn does NOT switch.
        elif is_setup_phase:
             # During setup, turn switches if player confirms ready OR if player passes while waiting
             if action_executed and (action == ACTION_SETUP_CONFIRM_READY or action == ACTION_PASS):
                 # Only switch if the *other* player isn't also ready (or just became ready)
                 if not (player.setup_ready and opponent.setup_ready):
                      switch_actual_turn = True
                      print("(Setup phase) Switching player...")
             elif not action_executed: # Failed setup action might forfeit turn? Let's assume it doesn't for now.
                 pass


        # --- Perform Turn Switch and Get Next State ---
        if switch_actual_turn and not done:
            print(f"--- End of {player.name}'s Actions ---")
            # TODO: End-of-turn checks (Poison, Burn, etc.) before switching
            self.game_state.switch_turn() # Increments turn number if needed
            self._start_turn() # Draw card, update energy stand for the new player
            # Check for deck out immediately after starting the new turn
            new_player = self.game_state.get_current_player()
            if not new_player.deck and not self.game_state.winner:
                opponent_player = self.game_state.get_opponent()
                print(f"Game Over! {new_player.name} decked out at start of turn! Winner: {opponent_player.name}")
                self.game_state.winner = opponent_player
                self.game_state.winner = opponent_player
                done = True
                if self.game_state.winner == player: reward += 2.0 # Opponent decked out
                else: reward -= 2.0 # Player decked out

        # Get the state representation for the player whose turn it is NOW
        next_player = self.game_state.get_current_player()
        next_state = self.get_state_representation(next_player)

        # Return the state for the *next* player, the reward obtained by the *acting* player, and done flag
        return next_state, reward, done


    def _commit_setup_choices(self):
        """Simultaneously moves pending setup cards from hand to active/bench for both players."""
        print("\n--- Committing Setup Choices ---")
        players_to_commit = [self.player1, self.player2]
        all_cards_to_remove_from_hand = {p.name: [] for p in players_to_commit}

        # 1. Assign Active Pokemon
        valid_setup = True
        for player in players_to_commit:
            if player.pending_setup_active and isinstance(player.pending_setup_active, PokemonCard):
                player.active_pokemon = player.pending_setup_active
                player.active_pokemon.is_active = True
                all_cards_to_remove_from_hand[player.name].append(player.pending_setup_active)
                print(f"{player.name} reveals Active: {player.active_pokemon.name}")
            else:
                print(f"CRITICAL ERROR: Player {player.name} has no valid pending active during commit!")
                valid_setup = False # Abort or handle error state

        if not valid_setup: return # Stop commit if error

        # 2. Assign Bench Pokemon
        for player in players_to_commit:
            valid_bench = []
            for card in player.pending_setup_bench:
                 if isinstance(card, PokemonCard): # Ensure they are Pokemon
                     valid_bench.append(card)
                     all_cards_to_remove_from_hand[player.name].append(card)
                 else:
                      print(f"Warning: Invalid card found in {player.name}'s pending bench during commit: {card}")

            player.bench = valid_bench # Assign only valid Pokemon
            for bench_pokemon in player.bench:
                bench_pokemon.is_active = False
            bench_names = ", ".join(p.name for p in player.bench) if player.bench else "None"
            print(f"{player.name} reveals Bench ({len(player.bench)}/{MAX_BENCH_SIZE}): [{bench_names}]")


        # 3. Remove all chosen cards from hands simultaneously using instance comparison
        for player in players_to_commit:
            cards_to_remove_instances = all_cards_to_remove_from_hand[player.name]
            original_hand = list(player.hand)
            new_hand = []
            removed_indices = set() # Track indices removed from original hand

            # Iterate through original hand and identify cards to keep
            for i, card_in_hand in enumerate(original_hand):
                is_card_to_remove = False
                # Check if this card instance matches any instance marked for removal
                for card_to_remove in cards_to_remove_instances:
                    if card_in_hand is card_to_remove: # Direct instance comparison
                        is_card_to_remove = True
                        removed_indices.add(i)
                        break # Found match, no need to check further for this hand card
                if not is_card_to_remove:
                    new_hand.append(card_in_hand)

            # Verify counts (optional debug check)
            if len(removed_indices) != len(cards_to_remove_instances):
                 print(f"Warning: Mismatch removing setup cards from {player.name}'s hand. Expected {len(cards_to_remove_instances)}, Removed {len(removed_indices)}.")
                 # This might happen if card instances were somehow duplicated or lost track of.
                 # The new_hand based on instance matching is likely the best effort.

            player.hand = new_hand # Assign the filtered hand
            print(f"{player.name} Hand size after setup commit: {len(player.hand)}")


        # 4. Clear pending attributes
        for player in players_to_commit:
            player.pending_setup_active = None
            player.pending_setup_bench = []
            # player.setup_ready remains True

        # 5. Reset current player to the designated starting player
        self.game_state.current_player_index = self.game_state.starting_player_index
        print(f"--- Setup Commit Complete (Current player: {self.game_state.get_current_player().name}) ---")


    def _execute_ability_effect(self, effect_tag: Optional[str], player: Player, opponent: Player, source_pokemon: PokemonCard) -> Tuple[bool, bool]:
        """
        Executes the game logic for a given ability effect tag.
        Returns (effect_executed: bool, ends_turn: bool)
        (Assumes the implementation from the original prompt is correct)
        """
        if not effect_tag: return False, False
        print(f"  Executing ability effect: {effect_tag}")
        ends_turn = False
        executed = False

        if effect_tag == "ABILITY_NIGHTMARE_AURA_WHENEVER_YOU_ATTACH_A_D_ENERGY_FROM_YOUR":
            print(f"  (Nightmare Aura is passive, triggered on attach, not used actively)")
            return False, ends_turn # Cannot be actively used

        elif effect_tag == "ABILITY_BROKEN_SPACE_BELLOW_ONCE_DURING_YOUR_TURN_":
            if source_pokemon.name == "Giratina ex":
                print(f"  {source_pokemon.name} uses Broken-Space Bellow, attaching Psychic energy.")
                source_pokemon.attach_energy("Psychic")
                ends_turn = True
                executed = True
            else: print(f"  Error: Broken-Space Bellow triggered by non-Giratina?")

        # Add other ability effects here...
        # elif effect_tag == "SOME_OTHER_ABILITY_TAG":
        #    # ... logic ...
        #    executed = True
        #    ends_turn = False # Or True if it ends turn

        else:
            print(f"  Warning: Unhandled ability effect tag '{effect_tag}'")

        return executed, ends_turn


    def _execute_trainer_effect(self, effect_tag: Optional[str], player: Player, opponent: Player,
                                target_bench_index: Optional[int] = None,
                                target_id: Optional[str] = None,
                                source_bench_index: Optional[int] = None) -> bool:
        """
        Executes the game logic for a given trainer card effect tag.
        Returns True if effect executed successfully, False otherwise.
        (Assumes the implementation from the original prompt is correct and handles various tags)
        """
        if not effect_tag: return False
        print(f"  Executing trainer effect: {effect_tag}")
        executed = False

        # --- Supporter Effects ---
        if effect_tag == "TRAINER_SUPPORTER_PROFESSOR_S_RESEARCH_DRAW_2_CARD":
            count = player.draw_cards(2)
            print(f"  Drew {count} cards.")
            executed = True # Drawing cards always considered successful unless deck empty handled in draw_cards
        elif effect_tag == "TRAINER_SUPPORTER_RED_DURING_THIS_TURN_ATTACKS_USE":
            print(f"  Red's effect active: +20 damage this turn.")
            self.actions_this_turn["red_effect_active"] = True
            executed = True
        elif effect_tag == "TRAINER_SUPPORTER_LEAF_DURING_THIS_TURN_THE_RETREA":
             print(f"  Leaf's effect active: Retreat cost -2 this turn.")
             self.actions_this_turn["leaf_effect_active"] = True
             executed = True
        elif effect_tag == "TRAINER_SUPPORTER_SABRINA_SWITCH_OUT_YOUR_OPPONENT":
            if opponent.active_pokemon and any(p and not p.is_fainted for p in opponent.bench):
                benched_indices = [i for i, p in enumerate(opponent.bench) if p and not p.is_fainted]
                # In real game, opponent chooses. Simulate random choice.
                chosen_bench_index = random.choice(benched_indices)
                new_active = opponent.bench.pop(chosen_bench_index) # Remove from bench first
                old_active = opponent.active_pokemon
                opponent.active_pokemon = new_active
                opponent.active_pokemon.is_active = True
                # Add old active to bench if space allows
                if opponent.can_place_on_bench():
                     opponent.place_on_bench(old_active)
                     old_active.is_active = False
                     print(f"  Sabrina forces switch. Opponent promotes {new_active.name}, benches {old_active.name}.")
                else: # Bench full, discard old active
                     opponent.discard_pile.append(old_active)
                     print(f"  Sabrina forces switch. Opponent promotes {new_active.name}. Old active {old_active.name} discarded (bench full).")
                executed = True
            else: print(f"  Cannot use Sabrina: No valid active or bench for opponent.")
        elif effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
            if opponent.active_pokemon and target_bench_index is not None:
                 if 0 <= target_bench_index < len(opponent.bench):
                     target_pokemon = opponent.bench[target_bench_index]
                     # Check if target is valid (exists, not fainted, damaged)
                     if target_pokemon and not target_pokemon.is_fainted and target_pokemon.current_hp < target_pokemon.hp:
                          new_active = opponent.bench.pop(target_bench_index)
                          old_active = opponent.active_pokemon
                          opponent.active_pokemon = new_active
                          opponent.active_pokemon.is_active = True
                          if opponent.can_place_on_bench():
                              opponent.place_on_bench(old_active)
                              old_active.is_active = False
                              print(f"  Cyrus switches opponent's {old_active.name} with benched {new_active.name} (Index {target_bench_index}).")
                          else:
                              opponent.discard_pile.append(old_active)
                              print(f"  Cyrus switches opponent's active with benched {new_active.name} (Index {target_bench_index}). Old active {old_active.name} discarded.")
                          executed = True
                     else: print(f"  Cannot use Cyrus: Target bench {target_bench_index} is not a valid damaged Pokemon.")
                 else: print(f"  Cannot use Cyrus: Invalid target bench index {target_bench_index}.")
            else: print(f"  Cannot use Cyrus: Opponent has no active or target index missing.")
        elif effect_tag == "TRAINER_SUPPORTER_MARS_YOUR_OPPONENT_SHUFFLES_THEI":
            print(f"  Mars forces opponent shuffle.")
            opp_hand_size = len(opponent.hand)
            opponent.deck.extend(opponent.hand)
            opponent.hand = []
            random.shuffle(opponent.deck)
            points_needed = max(0, POINTS_TO_WIN - opponent.points)
            count = opponent.draw_cards(points_needed)
            print(f"  Opponent shuffled {opp_hand_size} cards, drew {count} (targeting {points_needed}).")
            executed = True
        elif effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
            target_pokemon: Optional[PokemonCard] = None
            target_location = "invalid"
            if target_id == "active" and player.active_pokemon and player.active_pokemon.current_hp < player.active_pokemon.hp:
                target_pokemon = player.active_pokemon
                target_location = "active"
            elif target_id and target_id.startswith("bench_"):
                try:
                    bench_idx = int(target_id.split('_')[-1])
                    if 0 <= bench_idx < len(player.bench) and player.bench[bench_idx] and player.bench[bench_idx].current_hp < player.bench[bench_idx].hp:
                        target_pokemon = player.bench[bench_idx]
                        target_location = f"bench {bench_idx}"
                except (ValueError, IndexError): pass

            if target_pokemon:
                heal_amount = 30
                healed = min(heal_amount, target_pokemon.hp - target_pokemon.current_hp)
                target_pokemon.current_hp += healed
                # TODO: Remove special conditions if implemented
                print(f"  P.C. Lady healed {target_location} {target_pokemon.name} by {healed} HP.")
                executed = True
            else: print(f"  Cannot use P.C. Lady: Invalid or undamaged target specified ('{target_id}').")
        elif effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
            target = player.active_pokemon
            source_pokemon: Optional[PokemonCard] = None
            if source_bench_index is not None and 0 <= source_bench_index < len(player.bench):
                 potential_source = player.bench[source_bench_index]
                 if potential_source and not potential_source.is_fainted and potential_source.attached_energy:
                     source_pokemon = potential_source

            if target and not target.is_fainted and source_pokemon:
                 # Move one random energy
                 energy_type_to_move = random.choice(list(source_pokemon.attached_energy.keys()))
                 source_pokemon.attached_energy[energy_type_to_move] -= 1
                 if source_pokemon.attached_energy[energy_type_to_move] == 0:
                     del source_pokemon.attached_energy[energy_type_to_move]
                 target.attach_energy(energy_type_to_move)
                 print(f"  Dawn moved 1 {energy_type_to_move} from bench {source_bench_index} ({source_pokemon.name}) to active ({target.name}).")
                 executed = True
            else: print(f"  Cannot use Dawn: Invalid source ({source_bench_index}) or target specified.")
        elif effect_tag == "TRAINER_SUPPORTER_TEAM_ROCKET_GRUNT_FLIP_A_COIN_UN":
            heads_count = 0
            while True:
                if random.choice([True, False]): heads_count += 1; print("  Coin: Heads")
                else: print("  Coin: Tails"); break
            if opponent.active_pokemon and not opponent.active_pokemon.is_fainted:
                discarded_count = opponent.active_pokemon.discard_energy(heads_count) # Discard random
                print(f"  TR Grunt discarded {discarded_count} energy from opponent's active.")
                executed = True
            else: print("  Cannot use TR Grunt: Opponent has no valid active.")
        elif effect_tag == "TRAINER_SUPPORTER_IONO_EACH_PLAYER_SHUFFLES_THE_CA":
            print(f"  Iono used.")
            p_hand_size = len(player.hand)
            player.deck.extend(player.hand); player.hand = []; random.shuffle(player.deck)
            p_drew = player.draw_cards(p_hand_size)
            o_hand_size = len(opponent.hand)
            opponent.deck.extend(opponent.hand); opponent.hand = []; random.shuffle(opponent.deck)
            o_drew = opponent.draw_cards(o_hand_size)
            print(f"  Player shuffled {p_hand_size}, drew {p_drew}. Opponent shuffled {o_hand_size}, drew {o_drew}.")
            executed = True

        # --- Item Effects ---
        elif effect_tag == "TRAINER_ITEM_POKÉ_BALL_PUT_1_RANDOM_BASIC_POKEMON_":
            basic_in_deck = [card for card in player.deck if isinstance(card, PokemonCard) and card.is_basic]
            if basic_in_deck:
                chosen_pokemon = random.choice(basic_in_deck)
                player.deck.remove(chosen_pokemon)
                player.hand.append(chosen_pokemon)
                print(f"  Poke Ball found {chosen_pokemon.name}.")
                if len(player.hand) > MAX_HAND_SIZE: # Check hand size limit
                     # TCG Pocket rule: If hand limit exceeded by effect, discard happens immediately? Or choice?
                     # Simple approach: discard the drawn card if over limit
                     print(f"  Hand size limit ({MAX_HAND_SIZE}) exceeded. Discarding {chosen_pokemon.name}.")
                     player.hand.remove(chosen_pokemon) # Remove the card just added
                     player.discard_pile.append(chosen_pokemon)
                     # Alternative: prompt user/AI to discard - more complex
            else: print("  Poke Ball found no Basic Pokemon.")
            random.shuffle(player.deck) # Shuffle anyway
            executed = True
        elif effect_tag == "TRAINER_ITEM_POTION_HEAL_20_DAMAGE_FROM_1_OF_YOUR_":
             target_pokemon: Optional[PokemonCard] = None
             target_location = "invalid"
             if target_id == "active" and player.active_pokemon and player.active_pokemon.current_hp < player.active_pokemon.hp:
                 target_pokemon = player.active_pokemon
                 target_location = "active"
             elif target_id and target_id.startswith("bench_"):
                 try:
                     bench_idx = int(target_id.split('_')[-1])
                     if 0 <= bench_idx < len(player.bench) and player.bench[bench_idx] and player.bench[bench_idx].current_hp < player.bench[bench_idx].hp:
                         target_pokemon = player.bench[bench_idx]
                         target_location = f"bench {bench_idx}"
                 except (ValueError, IndexError): pass

             if target_pokemon:
                 heal_amount = 20
                 healed = min(heal_amount, target_pokemon.hp - target_pokemon.current_hp)
                 target_pokemon.current_hp += healed
                 print(f"  Potion healed {target_location} {target_pokemon.name} by {healed} HP.")
                 executed = True
             else: print(f"  Cannot use Potion: Invalid or undamaged target specified ('{target_id}').")

         # --- Tool Effects (Applied on attach/damage, just acknowledge play here) ---
        elif effect_tag == "TRAINER_TOOL_GIANT_CAPE_THE_POK_MON_THIS_CARD_IS_ATTACHED_TO_G":
            print(f"  (Giant Cape effect applied on attach)")
            executed = True # Playing the tool card is the action here
        elif effect_tag == "TRAINER_TOOL_ROCKY_HELMET_IF_THE_POKÉMON_THIS_CARD":
            print(f"  (Rocky Helmet effect applies when attacked)")
            executed = True # Playing the tool card is the action here

        # --- Fallback ---
        else:
            print(f"  Warning: Unhandled trainer effect tag '{effect_tag}'")
            # Assume the card is still played and discarded, but effect does nothing
            executed = False # Or True if playing the card itself is 'success'? Let's say False if effect unknown.

        return executed


# --- Helper: Dummy AI Agent ---
class DummyAIAgent:
    """A simple AI that chooses a random valid action."""
    def choose_action(self, state: Dict[str, Any], possible_actions: List[str]) -> str:
        if not possible_actions:
            return ACTION_PASS # Should not happen in a valid game state normally

        # Simple strategy: prioritize attacking if possible
        attack_actions = [a for a in possible_actions if a.startswith(ACTION_ATTACK_PREFIX)]
        if attack_actions:
            return random.choice(attack_actions)

        # Avoid passing if other actions exist
        non_pass_actions = [a for a in possible_actions if a != ACTION_PASS]
        if non_pass_actions:
            return random.choice(non_pass_actions)
        else:
            # Only PASS is possible
            return ACTION_PASS

# --- NEW Class for Human vs AI Gameplay ---
class HumanVsAI_Game(Game):
    def __init__(self, player1_deck_names: List[str], player2_deck_names: List[str],
                 player1_energy_types: List[str], player2_energy_types: List[str],
                 human_player_index: int, # 0 for Player 1, 1 for Player 2
                 ai_agent: Any): # Your trained AI model/agent object/function
        super().__init__(player1_deck_names, player2_deck_names, player1_energy_types, player2_energy_types)
        self.human_player_index = human_player_index
        self.ai_agent = ai_agent
        self.human_player = self.player1 if human_player_index == 0 else self.player2
        self.ai_player = self.player2 if human_player_index == 0 else self.player1

    def _display_pokemon_short(self, pokemon: Optional[PokemonCard], location: str = "Active") -> str:
        """Formats a short summary of a Pokemon."""
        if not pokemon:
            return f"{location}: <Empty>"
        tool_name = f" [{pokemon.attached_tool.name}]" if pokemon.attached_tool else ""
        energy_str = ", ".join(f"{e}:{c}" for e, c in pokemon.attached_energy.items()) if pokemon.attached_energy else "None"
        status = " FNT" if pokemon.is_fainted else ""
        return f"{location}: {pokemon.name} ({pokemon.current_hp}/{pokemon.hp} HP){tool_name}{status} E:[{energy_str}]"

    def _display_state_to_human(self):
        """Prints the game state in a human-readable format."""
        human = self.human_player
        ai = self.ai_player
        current_player = self.game_state.get_current_player()

        print("\n" + "="*40)
        print(f"--- {current_player.name}'s Turn ({self.game_state.turn_number}) ---")
        print("="*40)

        # --- Opponent (AI) State (Public Info) ---
        print(f"\n--- {ai.name}'s Field ---")
        print(f"Points: {ai.points}/{POINTS_TO_WIN}")
        print(f"Hand: {len(ai.hand)} cards")
        print(f"Deck: {len(ai.deck)} cards")
        print(f"Discard: {len(ai.discard_pile)} cards ({', '.join(c.name for c in ai.discard_pile[-5:])}{'...' if len(ai.discard_pile)>5 else ''})") # Show last 5 discarded
        print(self._display_pokemon_short(ai.active_pokemon, "Active"))
        print(f"Bench ({sum(1 for p in ai.bench if p)}/{MAX_BENCH_SIZE}):")
        for i, p in enumerate(ai.bench):
            if p:
                print(f"  B{i}: {p.name} ({p.current_hp}/{p.hp} HP){' [Tool]' if p.attached_tool else ''} E:[{sum(p.attached_energy.values())}]")
            # else: print(f"  B{i}: <Empty>") # Don't show empty slots for opponent
        print(f"Energy Stand: Preview={ai.energy_stand_preview or 'N/A'}, Available={'Yes' if ai.energy_stand_available else 'No'}")

        print("-" * 40) # Separator

        # --- Player (Human) State ---
        print(f"\n--- Your ({human.name}) Field ---")
        print(f"Points: {human.points}/{POINTS_TO_WIN}")
        print(f"Deck: {len(human.deck)} cards")
        print(f"Discard: {len(human.discard_pile)} cards ({', '.join(c.name for c in human.discard_pile[-5:])}{'...' if len(human.discard_pile)>5 else ''})")
        print(self._display_pokemon_short(human.active_pokemon, "Active"))
        print(f"Bench ({len(human.bench)}/{MAX_BENCH_SIZE}):")
        for i, p in enumerate(human.bench):
             print(f"  B{i}: {self._display_pokemon_short(p, '')}" if p else f"  B{i}: <Empty>")
        print(f"Energy Stand: Preview={human.energy_stand_preview or 'N/A'}, Available={human.energy_stand_available or 'None'}")

        print("\nYour Hand:")
        if not human.hand:
            print(" <Empty>")
        else:
            for i, card in enumerate(human.hand):
                # Check card type before accessing attributes
                if isinstance(card, PokemonCard):
                    # PokemonCards don't have card_type, use pokemon_type or just "Pokemon"
                    card_info = f"Pokemon - {card.pokemon_type}"
                elif hasattr(card, 'card_type'): # Check if TrainerCard or similar
                    card_info = card.card_type
                else:
                    card_info = "Unknown Type" # Fallback
                print(f"  H{i}: {card.name} ({card_info})")

        print("="*40 + "\n")


    def _get_human_action(self, possible_actions: List[str]) -> str:
        """Displays possible actions and gets validated input from the human."""
        print("--- Possible Actions ---")
        if not possible_actions:
            print("No actions possible? This might be an error. Forcing PASS.")
            return ACTION_PASS

        for i, action in enumerate(possible_actions):
            # Improve readability of actions
            action_desc = action.replace("ACTION_", "").replace("_", " ").title()
            if action.startswith(ACTION_PLAY_BASIC_BENCH_PREFIX):
                try: hand_idx = int(action.split('_')[-1]); action_desc += f" ({self.human_player.hand[hand_idx].name})"
                except: pass
            elif action.startswith(ACTION_ATTACH_ENERGY_BENCH_PREFIX):
                 try: bench_idx = int(action.split('_')[-1]); action_desc += f" ({self.human_player.bench[bench_idx].name})"
                 except: pass
            elif action.startswith(ACTION_ATTACK_PREFIX):
                 try: attack_idx = int(action.split('_')[-1]); action_desc += f" ({self.human_player.active_pokemon.attacks[attack_idx].name})"
                 except: pass
            elif action.startswith(ACTION_PLAY_SUPPORTER_PREFIX) or action.startswith(ACTION_PLAY_ITEM_PREFIX) or action.startswith(ACTION_ATTACH_TOOL_ACTIVE):
                 try: hand_idx = int(action.split('_')[-1]); action_desc += f" ({self.human_player.hand[hand_idx].name})"
                 except: pass # Handle complex prefixes below
            elif action.startswith(ACTION_ATTACH_TOOL_BENCH_PREFIX):
                 try: parts = action.split('_'); bench_idx, hand_idx = int(parts[-2]), int(parts[-1]); action_desc += f" (Card: {self.human_player.hand[hand_idx].name}, Target: Bench {bench_idx})"
                 except: pass
            elif action.startswith(ACTION_RETREAT_TO_BENCH_PREFIX):
                 try: bench_idx = int(action.split('_')[-1]); action_desc += f" (Bring in: {self.human_player.bench[bench_idx].name})"
                 except: pass
            # Add more specific descriptions for targeted actions (Potion, Cyrus, etc.)
            elif action.startswith(ACTION_PLAY_ITEM_POTION_TARGET_PREFIX):
                 try: parts = action.replace(ACTION_PLAY_ITEM_POTION_TARGET_PREFIX,"").split('_'); hand_idx=int(parts[0]); target=parts[1:]; action_desc = f"Play Potion ({self.human_player.hand[hand_idx].name}) on {' '.join(target).title()}"
                 except: pass
            elif action.startswith(ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX):
                 try: parts = action.replace(ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX,"").split('_'); hand_idx=int(parts[0]); bench_idx=int(parts[1]); action_desc = f"Play Cyrus ({self.human_player.hand[hand_idx].name}) targeting Opponent Bench {bench_idx}"
                 except: pass
            elif action.startswith(ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX) or action.startswith(ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX):
                 try: hand_idx = int(action.split('_')[-1]); action_desc += f" ({self.human_player.hand[hand_idx].name})"
                 except: pass


            print(f"  {i+1}: {action_desc}  ({action})") # Show both friendly and raw action name

        while True:
            try:
                choice = input(f"Choose action (1-{len(possible_actions)}): ")
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(possible_actions):
                    chosen_action = possible_actions[choice_index]
                    print(f"You chose: {chosen_action}")
                    return chosen_action
                else:
                    print("Invalid choice number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError: # Handle Ctrl+D or premature stream closing
                 print("\nInput terminated. Defaulting to PASS.")
                 return ACTION_PASS


    def play_game(self):
        """Runs the main game loop for Human vs AI."""
        done = False
        state = self.get_state_representation(self.game_state.get_current_player())  # Initial state

        while not done:
            current_player_obj = self.game_state.get_current_player()
            player_index = 0 if current_player_obj == self.player1 else 1

            # Check for win condition before player acts (e.g., deck out at start of turn)
            # Use check_win_condition() instead of direct attribute access
            winner_at_turn_start = self.game_state.check_win_condition()
            if winner_at_turn_start:
                 print(f"\nGame Over! Winner determined at start of turn: {winner_at_turn_start.name}")
                 done = True
                 break

            possible_actions = self.get_possible_actions()

            # Check if only PASS is possible (might indicate a stalemate or error)
            if len(possible_actions) == 1 and possible_actions[0] == ACTION_PASS:
                # Auto-pass if it's the only option, maybe add a small delay?
                print(f"{current_player_obj.name} has only PASS action available. Auto-passing.")
                action_to_take = ACTION_PASS
                # import time; time.sleep(1) # Optional delay
            elif player_index == self.human_player_index:
                # Human's turn
                self._display_state_to_human()
                action_to_take = self._get_human_action(possible_actions)
            else:
                # AI's turn
                print(f"\n--- {current_player_obj.name}'s (AI) Turn ---")
                print(f"AI thinking...")
                # Ensure the state sent to AI is from its perspective
                ai_state_view = self.get_state_representation(current_player_obj)
                action_to_take = self.ai_agent.choose_action(ai_state_view, possible_actions)
                # Validate AI action slightly
                if action_to_take not in possible_actions:
                    print(f"Warning: AI chose invalid action '{action_to_take}'. Valid: {possible_actions}. Defaulting to PASS.")
                    action_to_take = ACTION_PASS # Force PASS if AI messes up
                else:
                    print(f"AI chose action: {action_to_take}")
                # import time; time.sleep(1) # Optional delay to simulate thinking

            # Execute the chosen action using the parent class's step method
            try:
                # The state returned by step() is for the *next* player
                next_player_state, reward, done = self.step(action_to_take)
                print(f"Action resulted in reward: {reward:.3f}, Done: {done}") # Optional reward info
                # The state variable is updated implicitly by the game logic within step()
                # We don't strictly need to use 'next_player_state' in this loop structure
            except Exception as e:
                 print("\n!!! UNEXPECTED ERROR DURING GAME STEP !!!")
                 print(f"Action attempted: {action_to_take}")
                 print(f"Error: {e}")
                 traceback.print_exc()
                 print("Game cannot continue.")
                 done = True # End game on critical error


        # Game Over
        print("\n" + "="*40)
        print("--- GAME OVER ---")
        # Check win condition properly at the end
        winner = self.game_state.check_win_condition()
        if winner:
            print(f"Winner: {winner.name}!")
        elif self.game_state.turn_number > self.turn_limit:
             # Redetermine winner based on points if turn limit was the cause
             # Note: check_win_condition might not handle turn limit wins, so this logic is kept
             if self.player1.points > self.player2.points: winner_name = self.player1.name
             elif self.player2.points > self.player1.points: winner_name = self.player2.name
             else: winner_name = "Draw"
             print(f"Turn Limit Reached. Final Score -> P1: {self.player1.points}, P2: {self.player2.points}")
             if winner_name != "Draw": print(f"Winner by Points: {winner_name}")
             else: print("Result: Draw")
        else:
            print("Game ended without a clear winner condition met (potential error).")
            print(f"Final Score -> P1: {self.player1.points}, P2: {self.player2.points}")
        print("="*40)


# --- Example Usage ---
if __name__ == "__main__":
    # --- Constants and Helpers Copied/Adapted from rl_env.py for Prediction ---
    # Import necessary components from rl_env (ensure rl_env.py is accessible)
    try:
        from rl_env import (
            MAX_HAND_SIZE, MAX_BENCH_SIZE, CARD_TO_ID, ENERGY_TO_ID, UNKNOWN_CARD_ID, NO_ENERGY_ID,
            POKEMON_OBS_SIZE, OPP_POKEMON_OBS_SIZE, FLATTENED_OBS_SIZE, MAX_ATTACKS_PER_POKEMON,
            NUM_ENERGY_TYPES, ID_TO_ACTION, ACTION_MAP # Need ACTION_MAP for ID_TO_ACTION
        )
        print("Successfully imported constants from rl_env.py")
    except ImportError as e:
        print(f"Error importing from rl_env.py: {e}")
        print("Ensure rl_env.py is in the Python path or the same directory.")
        # Define fallback constants if import fails (less ideal, prone to mismatch)
        print("Using fallback constants - WARNING: May be outdated!")
        MAX_HAND_SIZE = 10
        MAX_BENCH_SIZE = 3
        # ... (Define ALL required constants here as a fallback) ...
        # This is complex and error-prone, fixing the import is better.
        exit() # Exit if constants cannot be loaded

    # Helper function copied from rl_env.py
    def _get_pokemon_state_dict_pred(pokemon: Optional[PokemonCard]) -> Optional[Dict[str, Any]]:
        """ Extracts relevant data from a PokemonCard object into a dictionary format. """
        if pokemon is None or pokemon.is_fainted:
            return None
        return {
            "name": pokemon.name,
            "current_hp": pokemon.current_hp,
            "max_hp": pokemon.hp,
            "energy": pokemon.attached_energy.copy(),
            "is_ex": pokemon.is_ex,
            "type": pokemon.pokemon_type,
            "weakness": pokemon.weakness_type,
            "can_attack": [pokemon.can_attack(i) for i in range(len(pokemon.attacks))],
            "num_attacks": len(pokemon.attacks)
        }

    # Helper function copied from rl_env.py
    def _format_pokemon_obs_pred(pokemon_state: Optional[Dict[str, Any]], include_can_attack=True) -> np.ndarray:
        """ Converts a Pokemon's state dictionary into a flat NumPy array segment. """
        target_size = POKEMON_OBS_SIZE if include_can_attack else OPP_POKEMON_OBS_SIZE
        if pokemon_state is None:
            return np.zeros(target_size, dtype=np.float32)

        card_id = float(CARD_TO_ID.get(pokemon_state["name"], UNKNOWN_CARD_ID))
        current_hp = float(pokemon_state["current_hp"])
        max_hp = float(pokemon_state["max_hp"])
        energy_counts = np.zeros(NUM_ENERGY_TYPES, dtype=np.float32)
        for type_str, count in pokemon_state["energy"].items():
            type_id = ENERGY_TO_ID.get(type_str, NO_ENERGY_ID)
            if 0 <= type_id < NUM_ENERGY_TYPES:
                 energy_counts[type_id] += float(count)
        is_ex = 1.0 if pokemon_state["is_ex"] else 0.0
        type_id = float(ENERGY_TO_ID.get(pokemon_state["type"], NO_ENERGY_ID))
        weakness_id = float(ENERGY_TO_ID.get(pokemon_state["weakness"], NO_ENERGY_ID))

        obs_list = [
            1.0, card_id, current_hp, max_hp, *energy_counts, is_ex, type_id, weakness_id,
        ]
        if include_can_attack:
            can_attack_flags = np.zeros(MAX_ATTACKS_PER_POKEMON, dtype=np.float32)
            num_actual_attacks = pokemon_state.get("num_attacks", 0)
            can_attack_list = pokemon_state.get("can_attack", [])
            for i in range(min(num_actual_attacks, MAX_ATTACKS_PER_POKEMON)):
                 if i < len(can_attack_list) and can_attack_list[i]:
                    can_attack_flags[i] = 1.0
            obs_list.extend(can_attack_flags)

        obs_array = np.array(obs_list, dtype=np.float32)
        if obs_array.shape[0] != target_size:
             raise ValueError(f"Prediction: Pokemon obs size mismatch! Expected {target_size}, got {obs_array.shape[0]}")
        return obs_array

    # Core observation formatting logic adapted from rl_env.py's _get_obs
    def _format_observation_for_prediction(state_dict: Dict[str, Any], current_player_obj: Player, opponent_obj: Player) -> np.ndarray:
        """ Constructs the flattened observation vector for model prediction. """
        # My State
        my_hand_ids = [CARD_TO_ID.get(name, UNKNOWN_CARD_ID) for name in state_dict["my_hand_cards"]]
        my_hand_padded = np.pad(my_hand_ids, (0, MAX_HAND_SIZE - len(my_hand_ids)), constant_values=UNKNOWN_CARD_ID).astype(np.float32)
        my_deck_size = np.array([state_dict["my_deck_size"]], dtype=np.float32)
        my_discard_size = np.array([state_dict["my_discard_size"]], dtype=np.float32)
        my_points = np.array([state_dict["my_points"]], dtype=np.float32)
        my_energy_available_id = np.array([ENERGY_TO_ID.get(state_dict["my_energy_stand_available"], NO_ENERGY_ID)], dtype=np.float32)
        my_energy_preview_id = np.array([ENERGY_TO_ID.get(state_dict["my_energy_stand_preview"], NO_ENERGY_ID)], dtype=np.float32)
        my_active_state_dict = _get_pokemon_state_dict_pred(current_player_obj.active_pokemon)
        my_active_flat = _format_pokemon_obs_pred(my_active_state_dict, include_can_attack=True)
        my_bench_flat_list = []
        for i in range(MAX_BENCH_SIZE):
            pokemon = current_player_obj.bench[i] if i < len(current_player_obj.bench) else None
            bench_poke_state_dict = _get_pokemon_state_dict_pred(pokemon)
            my_bench_flat_list.append(_format_pokemon_obs_pred(bench_poke_state_dict, include_can_attack=True))
        my_bench_flat = np.concatenate(my_bench_flat_list)

        # Opponent State
        opp_hand_size = np.array([state_dict["opp_hand_size"]], dtype=np.float32)
        opp_deck_size = np.array([state_dict["opp_deck_size"]], dtype=np.float32)
        opp_discard_size = np.array([state_dict["opp_discard_size"]], dtype=np.float32)
        opp_points = np.array([state_dict["opp_points"]], dtype=np.float32)
        opp_energy_available_exists = np.array([1.0 if state_dict["opp_energy_stand_status"]["available_exists"] else 0.0], dtype=np.float32)
        opp_energy_preview_id = np.array([ENERGY_TO_ID.get(state_dict["opp_energy_stand_status"]["preview"], NO_ENERGY_ID)], dtype=np.float32)
        opp_active_state_dict = _get_pokemon_state_dict_pred(opponent_obj.active_pokemon)
        opp_active_flat = _format_pokemon_obs_pred(opp_active_state_dict, include_can_attack=False)
        opp_bench_size = np.array([state_dict["opp_bench_size"]], dtype=np.float32)

        # Global State
        turn_number = np.array([state_dict["turn"]], dtype=np.float32)
        can_attach_energy = np.array([1.0 if state_dict["can_attach_energy"] else 0.0], dtype=np.float32)
        is_first_turn = np.array([1.0 if state_dict["is_first_turn"] else 0.0], dtype=np.float32)

        # Concatenate
        flat_obs = np.concatenate([
            my_hand_padded, my_deck_size, my_discard_size, my_points,
            my_energy_available_id, my_energy_preview_id, my_active_flat, my_bench_flat,
            opp_hand_size, opp_deck_size, opp_discard_size, opp_points,
            opp_energy_available_exists, opp_energy_preview_id, opp_active_flat, opp_bench_size,
            turn_number, can_attach_energy, is_first_turn,
        ])

        if flat_obs.shape[0] != FLATTENED_OBS_SIZE:
            raise ValueError(f"Prediction: Observation size mismatch! Expected {FLATTENED_OBS_SIZE}, got {flat_obs.shape[0]}.")

        return flat_obs.astype(np.float32)

    # --- End of Copied/Adapted Helpers ---


    # Make sure cards.json is accessible from where you run this script.
    # You might need to adjust the CARD_DATA_FILE path definition within the Game class
    # or place cards.json in the same directory as this script for the placeholder path to work.
    print(f"Attempting to load card data from: {CARD_DATA_FILE}")
    print(f"Current working directory: {os.getcwd()}")

    # --- Define Sample Decks ---
    # Replace with actual deck lists based on your cards.json names
    # Example decks (needs cards like Pikachu, Giratina ex, Darkrai ex, Prof Research etc. in cards.json)
    player1_deck_list = [
        "Giratina ex", "Giratina ex", # Basic Pokemon + Stage 1 (if implemented)
        "Darkrai ex", "Darkrai ex", # Supporters
        "Professor's Research", "Professor's Research", "Poké Ball", "Poké Ball", "Potion", "Potion", "Giant Cape", "Rocky Helmet", "Rocky Helmet", "Sabrina", "Leaf", "Mars", "Mars", "Red", "Cyrus", "Pokémon Center Lady"
    ]  # Repeat to get 20 cards (Adjust as needed)
    player1_deck_list = player1_deck_list[:20] # Ensure exactly 20

    player2_deck_list = [
        "Giratina ex", "Giratina ex", # Basic Pokemon + Stage 1 (if implemented)
        "Darkrai ex", "Darkrai ex", # Supporters
        "Professor's Research", "Professor's Research", "Poké Ball", "Poké Ball", "Potion", "Potion", "Giant Cape", "Rocky Helmet", "Rocky Helmet", "Sabrina", "Leaf", "Mars", "Mars", "Red", "Cyrus", "Pokémon Center Lady"
    ]
    player2_deck_list = player2_deck_list[:20] # Ensure exactly 20

    player1_energies = ["Darkness"] # Primary energy type(s)
    player2_energies = ["Darkness"]

    # --- Choose Human Player ---
    human_is_player = 1 # 1 or 2

    # --- Create AI Agent ---
    # Load the trained model
    try:
        from sb3_contrib import MaskablePPO
        MODEL_PATH = "models/pokemon_maskable_ppo_agent.zip"
        model = MaskablePPO.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and you have the correct libraries installed (stable-baselines3 / sb3-contrib).")
        exit()

    class TrainedAIAgent:
        def __init__(self, player_obj: Player, opponent_obj: Player):
             # Store references to the actual player objects needed for formatting
             self.player_obj = player_obj
             self.opponent_obj = opponent_obj

        def choose_action(self, state_dict: Dict[str, Any], possible_actions: List[str]) -> str:
            # Generate action mask (using imported ID_TO_ACTION)
            action_mask = np.array([1 if ID_TO_ACTION[i] in possible_actions else 0 for i in range(len(ID_TO_ACTION))], dtype=np.int8) # Use numpy array for SB3

            # Format the observation dictionary into a NumPy array
            # Pass the actual player objects to the formatting function
            obs = _format_observation_for_prediction(state_dict, self.player_obj, self.opponent_obj)

            # Predict action using the loaded model
            action_id, _states = model.predict(obs, deterministic=True, action_masks=action_mask)

            # Convert action ID back to string
            action = ID_TO_ACTION[action_id]
            return action

    # --- Initialize and Run Game ---
    try:
        # Determine player objects *before* creating the agent
        human_player_obj = None
        ai_player_obj = None
        temp_game_for_setup = HumanVsAI_Game( # Create temporary game to get player objects
             player1_deck_names=player1_deck_list,
             player2_deck_names=player2_deck_list,
             player1_energy_types=player1_energies,
             player2_energy_types=player2_energies,
             human_player_index=human_is_player - 1, # Use 0-based index
             ai_agent=None # No agent needed yet
        )
        if human_is_player == 1:
            human_player_obj = temp_game_for_setup.player1
            ai_player_obj = temp_game_for_setup.player2
        else:
            human_player_obj = temp_game_for_setup.player2
            ai_player_obj = temp_game_for_setup.player1

        # Now create the AI agent, passing the correct player objects
        ai_opponent = TrainedAIAgent(player_obj=ai_player_obj, opponent_obj=human_player_obj)

        # Adjust indices based on human_is_player choice
        if human_is_player == 1:
             game = HumanVsAI_Game(
                 player1_deck_names=player1_deck_list,
                 player2_deck_names=player2_deck_list,
                 player1_energy_types=player1_energies,
                 player2_energy_types=player2_energies,
                 human_player_index=human_is_player - 1, # Use 0-based index
                 ai_agent=ai_opponent # Pass the created agent
             )
        # Remove the redundant elif block, the single instantiation handles both cases
        else:
             raise ValueError("human_is_player must be 1 or 2")

        game.play_game()

    except ValueError as e:
         print(f"Error initializing game: {e}")
    except FileNotFoundError:
         print("ERROR: cards.json not found. Please ensure it's in the correct location.")
         print(f"Expected location based on script: {CARD_DATA_FILE}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         traceback.print_exc()
