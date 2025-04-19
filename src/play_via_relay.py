import sys
import random
import traceback
import numpy as np
from typing import List, Optional, Dict, Tuple, Any

# --- Game Simulator Imports ---
# Assuming play_vs_ai.py is runnable from a location that can find 'simulator'
try:
    from simulator.game import Game
    # Import action constants needed for parsing user input and mapping AI actions
    from simulator.game import (
        ACTION_PASS, ACTION_ATTACK_PREFIX, ACTION_ATTACH_ENERGY_ACTIVE,
        ACTION_ATTACH_ENERGY_BENCH_PREFIX, ACTION_PLAY_BASIC_BENCH_PREFIX,
        ACTION_USE_ABILITY_ACTIVE, ACTION_USE_ABILITY_BENCH_PREFIX,
        ACTION_PLAY_SUPPORTER_PREFIX, ACTION_PLAY_ITEM_PREFIX,
        ACTION_ATTACH_TOOL_ACTIVE, ACTION_ATTACH_TOOL_BENCH_PREFIX, ACTION_RETREAT_TO_BENCH_PREFIX,
        ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX, ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX,
        ACTION_SETUP_CONFIRM_READY, ACTION_OPP_PLAY_SUPPORTER_PREFIX, ACTION_OPP_PLAY_ITEM_PREFIX,
        ACTION_OPP_ATTACH_TOOL_PREFIX, ACTION_OPP_PLAY_BASIC_PREFIX,
        # Target-specific action prefixes (crucial for parsing)
        ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX,
        ACTION_PLAY_ITEM_POTION_TARGET_PREFIX,
        ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX,
        ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX
    )
    from simulator.entities import Player, PokemonCard, TrainerCard, MAX_BENCH_SIZE, POINTS_TO_WIN
    # Although we don't instantiate rl_env, we use its action mapping
    from rl_env import (
    ACTION_MAP, ID_TO_ACTION, PokemonTCGPocketEnv, # Original imports
    ARCHETYPE_CORE, TECH_POOL, CORE_SIZE, NUM_TECH_SLOTS # Add these constants
)  # For action mapping and example decks
except ImportError as e:
    print(f"Error importing simulator modules: {e}")
    print("Please ensure play_vs_ai.py is in the correct directory relative to the 'simulator' package,")
    print("or that the 'simulator' directory is in your PYTHONPATH.")
    sys.exit(1)

# --- Stable Baselines 3 & SB3-Contrib Imports ---
try:
    # from stable_baselines3 import PPO # No longer needed directly for loading
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv # Import DummyVecEnv
    from sb3_contrib import MaskablePPO # Import MaskablePPO
except ImportError as e:
    print(f"Error importing Stable Baselines 3 or SB3-Contrib modules: {e}")
    print("Please ensure stable-baselines3 and sb3-contrib are installed (`pip install stable-baselines3 sb3-contrib`).")
    sys.exit(1)


# --- Model Loading and Prediction ---

def load_my_model(model_path: str, vec_normalize_path: str) -> Tuple[Optional[MaskablePPO], Optional[VecNormalize]]: # Updated type hint
    """
    Loads the trained SB3 MaskablePPO model and the corresponding VecNormalize object.

    Args:
        model_path: Path to the SB3 model zip file.
        vec_normalize_path: Path to the VecNormalize pkl file.

    Returns:
        A tuple containing the loaded model and VecNormalize object, or (None, None) if loading fails.
    """
    print(f"--- Loading model from {model_path} ---")
    print(f"--- Loading VecNormalize from {vec_normalize_path} ---")
    try:
        custom_objects = {"policy_kwargs": {}}
        model = MaskablePPO.load(model_path, device='auto', custom_objects=custom_objects)
        dummy_env = DummyVecEnv([lambda: PokemonTCGPocketEnv()])
        vec_env = VecNormalize.load(vec_normalize_path, venv=dummy_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("--- Model and VecNormalize loaded successfully ---")
        return model, vec_env
    except Exception as e:
        print(f"--- Error loading model or VecNormalize: {e} ---")
        traceback.print_exc()
        return None, None

def predict_action(model: MaskablePPO, observation: np.ndarray, valid_actions_mask: np.ndarray) -> int: # Updated type hint
    """
    Gets an action prediction from the loaded SB3 MaskablePPO model.
    """
    action_id, _states = model.predict(observation, action_masks=valid_actions_mask, deterministic=True)
    return int(action_id)

# --- Helper Functions ---

def display_game_state(game: Game, perspective_player: Player, show_opponent_hand=False):
    """Prints a comprehensive view of the game state for the user."""
    print("\n" + "="*60)
    gs = game.game_state
    p1 = game.player1
    p2 = game.player2
    current_turn_player = gs.get_current_player()

    print(f"Turn: {gs.turn_number} | Current Player: {current_turn_player.name} ({'AI' if current_turn_player == perspective_player else 'Opponent'})")
    print(f"First Turn Overall? {'Yes' if gs.is_first_turn else 'No'}")
    print("-" * 20)

    ai_player_obj = perspective_player
    # Determine opponent relative to perspective_player for consistent labeling
    opponent_player_obj = game.player1 if perspective_player == game.player2 else game.player2

    # Display AI State
    print(f"AI ({ai_player_obj.name}):")
    print(f"  Points: {ai_player_obj.points}/{POINTS_TO_WIN}")
    print(f"  Hand: {len(ai_player_obj.hand)} cards")
    print(f"  AI Hand Cards:")
    for i, card in enumerate(ai_player_obj.hand):
        print(f"    [{i}] {card!r}")
    print(f"  Deck: {len(ai_player_obj.deck)} | Discard: {len(ai_player_obj.discard_pile)}")
    print(f"  Energy Stand: Avail='{ai_player_obj.energy_stand_available or 'None'}', Preview='{ai_player_obj.energy_stand_preview or 'None'}'")
    print(f"  Active: {ai_player_obj.active_pokemon!r}")
    bench_str = ", ".join(f"[{i}] {p!r}" for i, p in enumerate(ai_player_obj.bench)) if ai_player_obj.bench else "Empty"
    print(f"  Bench ({len(ai_player_obj.bench)}/{MAX_BENCH_SIZE}): {bench_str}")
    print("-" * 20)

    # Display Opponent State
    print(f"Opponent ({opponent_player_obj.name}):")
    print(f"  Points: {opponent_player_obj.points}/{POINTS_TO_WIN}")
    print(f"  Hand: {len(opponent_player_obj.hand)} cards")
    if show_opponent_hand:
        print(f"  Opponent Hand Cards (Debug):")
        for i, card in enumerate(opponent_player_obj.hand):
            print(f"    [{i}] {card!r}")
    print(f"  Deck: {len(opponent_player_obj.deck)} | Discard: {len(opponent_player_obj.discard_pile)}")
    opp_avail_str = 'Yes' if opponent_player_obj.energy_stand_available else 'No'
    print(f"  Energy Stand: Avail='{opp_avail_str}', Preview='{opponent_player_obj.energy_stand_preview or 'None'}'")
    print(f"  Active: {opponent_player_obj.active_pokemon!r}")
    bench_str = ", ".join(f"[{i}] {p!r}" for i, p in enumerate(opponent_player_obj.bench)) if opponent_player_obj.bench else "Empty"
    print(f"  Bench ({len(opponent_player_obj.bench)}/{MAX_BENCH_SIZE}): {bench_str}")
    print("="*60 + "\n")


def get_ai_turn_action(model: MaskablePPO, vec_env: VecNormalize, game: Game, ai_player: Player) -> str:
    """Gets state, normalizes it, gets mask, predicts, and returns the action string for the AI."""
    temp_env = PokemonTCGPocketEnv()
    temp_env.game = game
    temp_env.current_player = ai_player
    raw_observation = temp_env._get_obs()
    valid_mask = temp_env.action_mask_fn()
    temp_env.close()

    normalized_observation = vec_env.normalize_obs(np.array([raw_observation]))[0]
    action_id = predict_action(model, normalized_observation, valid_mask)
    action_str = ID_TO_ACTION.get(action_id)

    if action_str is None or not valid_mask[action_id]:
        original_action_str = action_str # Store for logging if it was invalid but not None
        if action_str is not None and not valid_mask[action_id]:
            print(f"--- WARNING: AI predicted action '{action_str}' (ID: {action_id}) which is NOT in the valid mask! ---")
        else:
            print(f"--- ERROR: AI model predicted invalid action ID: {action_id} (mapped to: {original_action_str}) ---")

        pass_action_id = ACTION_MAP.get(ACTION_PASS)
        if pass_action_id is not None and valid_mask[pass_action_id]:
            print("--- AI falling back to PASS ---")
            return ACTION_PASS
        else:
            valid_ids = np.where(valid_mask)[0]
            if len(valid_ids) > 0:
                fallback_id = valid_ids[0]
                fallback_str = ID_TO_ACTION.get(fallback_id)
                print(f"--- AI falling back to first valid action: {fallback_str} (ID: {fallback_id}) ---")
                return fallback_str
            else:
                print("--- CRITICAL ERROR: AI has no valid actions and cannot PASS. Returning PASS as last resort. ---")
                return ACTION_PASS # Should ideally not happen
    return action_str


def parse_opponent_action(input_str: str, game: Game, opponent_player: Player) -> Optional[str]:
    """Parses the user's input string into a game action string."""
    parts = input_str.strip().split()
    if not parts:
        return None

    action_type = parts[0].lower()
    args = parts[1:]

    is_setup_phase = not opponent_player.setup_ready
    if is_setup_phase:
        # Re-using player's hand for parsing opponent's setup name selection
        # This is a bit of a hack for relay, assuming the opponent has *some* card with that name in their deck.
        # The game logic for opponent setup actions will create "ghost" cards.
        if action_type == "active":
            if not args: print("Usage: active <Pokemon Card Name>"); return None
            card_name = " ".join(args)
            # Find any basic pokemon with this name in the opponent's hand (for index)
            # In relay, we don't *really* care about the index, game.step handles name.
            # We just need to construct a valid-looking action string.
            # For robustness, we'll check if the name is a valid basic Pokemon.
            if card_name not in game.card_data or not (game.card_data[card_name].get("card_type") == "Pokemon" and game.card_data[card_name].get("is_basic")):
                print(f"Error: '{card_name}' is not a recognized Basic Pokemon name."); return None
            # The actual index doesn't matter as much as game.step will use the card name for opponent.
            # However, to be consistent with how AI actions are structured (using an index from *their* hand)
            # let's find a pseudo-index. This is mostly for internal consistency of action string format.
            # The game.step will use the *name* for ghost card creation.
            pseudo_hand_index = 0 # Default, as real index is not used by opponent logic
            for i, card in enumerate(opponent_player.hand): # Iterate over opponent's *actual* hand for index
                 if isinstance(card, PokemonCard) and card.is_basic and card.name == card_name:
                     if opponent_player.pending_setup_bench and card in opponent_player.pending_setup_bench: continue
                     pseudo_hand_index = i
                     break
            return f"{ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX}{pseudo_hand_index}" # Pass name via index

        elif action_type == "bench":
            if not args: print("Usage: bench <Pokemon Card Name>"); return None
            card_name = " ".join(args)
            if card_name not in game.card_data or not (game.card_data[card_name].get("card_type") == "Pokemon" and game.card_data[card_name].get("is_basic")):
                print(f"Error: '{card_name}' is not a recognized Basic Pokemon name."); return None
            pseudo_hand_index = 0
            for i, card in enumerate(opponent_player.hand):
                 if isinstance(card, PokemonCard) and card.is_basic and card.name == card_name:
                     if card == opponent_player.pending_setup_active: continue
                     if opponent_player.pending_setup_bench and card in opponent_player.pending_setup_bench: continue
                     pseudo_hand_index = i
                     break
            if len(opponent_player.pending_setup_bench) >= MAX_BENCH_SIZE:
                print(f"Error: Bench is full."); return None
            return f"{ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX}{pseudo_hand_index}"

        elif action_type == "ready":
            if opponent_player.pending_setup_active is None: # This check still relies on sim state
                print("Error: Cannot be ready without selecting an active Pokemon first (in simulation).")
                # For relay, if user says ready, we assume they did it.
                # The game.step should validate if the opponent has a pending_active.
                # It's better if the user ensures they chose an active first.
            return ACTION_SETUP_CONFIRM_READY
        elif action_type == "pass":
            # Pass is only valid if opponent is ready and AI is not.
            ai_player = game.player1 if opponent_player == game.player2 else game.player2
            if opponent_player.setup_ready and not ai_player.setup_ready: return ACTION_PASS
            else: print("Error: Cannot pass during your setup phase unless you are ready and waiting for AI."); return None
        else:
            print("Invalid setup command. Use: active <Name>, bench <Name>, ready, or pass (if waiting).")
            return None

    # --- Normal Turn Actions ---
    if action_type == "pass": return ACTION_PASS
    elif action_type == "attack":
        try:
            attack_index = int(args[0])
            if attack_index >= 0: return f"{ACTION_ATTACK_PREFIX}{attack_index}"
            else: print("Error: Attack index must be non-negative.")
        except (IndexError, ValueError): print("Usage: attack <attack_index>")
        return None
    elif action_type == "energy":
        try:
            target = args[0].lower()
            if target == "active": return ACTION_ATTACH_ENERGY_ACTIVE
            elif target == "bench":
                bench_index = int(args[1])
                if 0 <= bench_index < MAX_BENCH_SIZE: return f"{ACTION_ATTACH_ENERGY_BENCH_PREFIX}{bench_index}"
                else: print(f"Error: Invalid bench index '{args[1]}'. Must be 0-{MAX_BENCH_SIZE-1}.")
            else: print("Usage: energy active | energy bench <index>")
        except (IndexError, ValueError): print("Usage: energy active | energy bench <index>")
        return None
    elif action_type == "ability":
        try:
            source = args[0].lower()
            if source == "active": return ACTION_USE_ABILITY_ACTIVE
            elif source == "bench":
                bench_index = int(args[1])
                if 0 <= bench_index < MAX_BENCH_SIZE: return f"{ACTION_USE_ABILITY_BENCH_PREFIX}{bench_index}"
                else: print(f"Error: Invalid bench index '{args[1]}'.")
            else: print("Usage: ability active | ability bench <index>")
        except (IndexError, ValueError): print("Usage: ability active | ability bench <index>")
        return None
    elif action_type == "retreat":
        try:
            bench_index = int(args[0])
            if 0 <= bench_index < MAX_BENCH_SIZE: return f"{ACTION_RETREAT_TO_BENCH_PREFIX}{bench_index}"
            else: print(f"Error: Invalid bench index '{args[0]}'.")
        except (IndexError, ValueError): print("Usage: retreat <target_bench_index>")
        return None
    elif action_type == "bench": # Play Basic Pokemon (from hand - ghost card)
        if not args: print("Usage: bench <Pokemon Card Name>"); return None
        card_name = " ".join(args)
        if card_name not in game.card_data: print(f"Error: Card name '{card_name}' not recognized."); return None
        card_info = game.card_data[card_name]
        if card_info.get("card_type") != "Pokemon" or not card_info.get("is_basic"):
            print(f"Error: '{card_name}' is not a Basic Pokemon."); return None
        card_name_hyphenated = card_name.replace(' ', '-')
        return f"{ACTION_OPP_PLAY_BASIC_PREFIX}{card_name_hyphenated}"
    elif action_type == "supporter" or action_type == "item" or action_type == "tool":
        card_name = ""
        target_info_parts = []
        current_name_parts = []
        found_valid_name = False
        for i, part in enumerate(args):
            current_name_parts.append(part)
            potential_name = " ".join(current_name_parts)
            if potential_name in game.card_data:
                card_info = game.card_data[potential_name]
                if card_info.get("card_type", "").lower() == action_type:
                    card_name = potential_name
                    target_info_parts = args[i+1:]
                    found_valid_name = True
                    break
        if not found_valid_name:
            print(f"Error: Could not find a valid {action_type.capitalize()} card name starting with '{' '.join(args)}'."); return None
        card_name_hyphenated = card_name.replace(' ', '-')
        action_prefix_map = {
            "supporter": ACTION_OPP_PLAY_SUPPORTER_PREFIX,
            "item": ACTION_OPP_PLAY_ITEM_PREFIX,
            "tool": ACTION_OPP_ATTACH_TOOL_PREFIX
        }
        action_str_base = f"{action_prefix_map[action_type]}{card_name_hyphenated}"
        if target_info_parts:
            target_type = target_info_parts[0].lower()
            target_index_str = target_info_parts[1] if len(target_info_parts) > 1 else None
            if target_type == "active": return f"{action_str_base}_active"
            elif target_type == "bench":
                if target_index_str is not None:
                    try:
                        bench_idx = int(target_index_str)
                        if 0 <= bench_idx < MAX_BENCH_SIZE: return f"{action_str_base}_bench_{bench_idx}"
                        else: print(f"Error: Invalid bench index '{bench_idx}'."); return None
                    except ValueError: print(f"Error: Bench index '{target_index_str}' must be a number."); return None
                else: print(f"Error: Missing bench index after 'bench'."); return None
            else: # For cards like Cyrus, target might be just index
                return f"{action_str_base}_{'_'.join(target_info_parts)}"
        else: return action_str_base
    else:
        print(f"Unknown action type: '{action_type}'")
        print("Valid actions: pass, attack, energy, bench, ability, supporter, item, tool, retreat")
        return None

def main():
    print("--- Pokemon TCG Pocket AI Relay Interface ---")

    MODEL_PATH = "models/pokemon_maskable_ppo_agent.zip"
    VEC_NORMALIZE_PATH = "models/vecnormalize.pkl"
    player1_deck_names = ["Darkrai ex"] * 2 + ["Giratina ex"] * 2 + ["Professor's Research"] * 2 + ["Poké Ball"] * 2 + ["Potion"] * 2 + ["Giant Cape"] + ["Rocky Helmet"] * 2 + ["Sabrina"] + ["Leaf"] + ["Mars"] * 2 + ["Red"] + ["Cyrus"] + ["Pokémon Center Lady"]
    random.shuffle(player1_deck_names)
    player1_deck_names = player1_deck_names[:20]
    player1_energy_types = ["Darkness"]

    player2_deck_names = list(ARCHETYPE_CORE)
    available_tech = list(TECH_POOL.items())
    random.shuffle(available_tech)
    current_tech_count = 0
    num_tech_slots = NUM_TECH_SLOTS
    for tech_card, max_count in available_tech:
        if current_tech_count < num_tech_slots:
            num_to_add = min(max_count, num_tech_slots - current_tech_count)
            if random.random() < 0.7:
                player2_deck_names.extend([tech_card] * num_to_add)
                current_tech_count += num_to_add
        else: break
    random.shuffle(player2_deck_names)
    player2_deck_names = player2_deck_names[:20]
    player2_energy_types = ["Darkness"]

    ai_model, vec_env = load_my_model(MODEL_PATH, VEC_NORMALIZE_PATH)
    if ai_model is None or vec_env is None: sys.exit(1)

    ai_player_num = -1
    while ai_player_num not in [1, 2]:
        try: ai_player_num = int(input("Which player is the AI? (1 or 2): ").strip())
        except ValueError: print("Invalid input.")
    first_player_str = ""
    while first_player_str not in ["ai", "opponent"]:
        first_player_str = input("Who goes first? (ai / opponent): ").lower().strip()

    try:
        game = Game(player1_deck_names, player2_deck_names, player1_energy_types, player2_energy_types)
    except Exception as e: print(f"FATAL ERROR during game initialization: {e}"); traceback.print_exc(); sys.exit(1)

    ai_player = game.player1 if ai_player_num == 1 else game.player2
    opponent_player = game.player2 if ai_player_num == 1 else game.player1
    ai_player.skip_automatic_draw = True
    opponent_player.skip_automatic_draw = False # Opponent draw is simulated by game.step()

    game.game_state.starting_player_index = ai_player_num - 1 if first_player_str == "ai" else 1 - (ai_player_num - 1)
    game.game_state.current_player_index = game.game_state.starting_player_index
    print(f"Game starts. Player {game.game_state.starting_player_index + 1} ({game.game_state.get_current_player().name}) goes first.")

    print("\n--- Manual AI Hand Input ---")
    hand_set_successfully = False
    while not hand_set_successfully:
        print(f"Simulated AI ({ai_player.name}) initial hand: {[c.name for c in ai_player.hand]}")
        user_hand_input = input("Enter AI's 5 starting cards (Comma-separated, exact names): ")
        target_card_names = [name.strip() for name in user_hand_input.split(',')]
        if len(target_card_names) != 5: print(f"Error: Enter 5 card names."); continue
        current_sim_hand = list(ai_player.hand); current_sim_deck = list(ai_player.deck)
        available_instances = current_sim_hand + current_sim_deck
        new_correct_hand_instances = []; found_flags = [False] * 5; not_found_names = []
        temp_available_instances = list(available_instances); valid_names = True
        for i, target_name in enumerate(target_card_names):
            if target_name not in game.card_data:
                print(f"Error: Card name '{target_name}' unrecognized."); not_found_names.append(f"{target_name} (Unknown)"); valid_names = False; continue
            found_instance_for_this_name = False
            for idx, instance in enumerate(temp_available_instances):
                if instance.name == target_name:
                    new_correct_hand_instances.append(temp_available_instances.pop(idx)); found_flags[i] = True; found_instance_for_this_name = True; break
            if not found_instance_for_this_name and valid_names: not_found_names.append(target_name)
        if not valid_names: print("Correct unknown names and try again."); continue
        if all(found_flags):
            new_correct_deck = temp_available_instances; random.shuffle(new_correct_deck)
            ai_player.hand = new_correct_hand_instances; ai_player.deck = new_correct_deck
            print("\n--- AI Hand Successfully Set ---"); hand_set_successfully = True
        else:
            print("\nError: Could not find instances for: " + ", ".join(not_found_names)); print("Check typos/deck list.");

    print("\n--- AI Starting Hand (Manually Set) ---")
    for i, card in enumerate(ai_player.hand): print(f"  [{i}] {card!r}")
    print("--------------------------------------")
    input("Press Enter when ready to start setup...")

    print("\n--- Entering Setup Phase ---")
    while not game.player1.setup_ready or not game.player2.setup_ready:
        current_setup_player = game.game_state.get_current_player()
        is_ai_turn_setup = (current_setup_player == ai_player)
        display_game_state(game, ai_player)
        if is_ai_turn_setup:
            print(f"--- AI's turn ({ai_player.name}) for setup ---")
            try:
                ai_action_str = get_ai_turn_action(ai_model, vec_env, game, ai_player)
                readable_action = get_readable_ai_action(ai_action_str, ai_player)
                print(f"\nAI chooses setup action: {readable_action}")
                input("Press Enter to apply AI's setup action...")
                _next_state_dict, _reward, _done = game.step(ai_action_str)
            except Exception as e: print(f"Error during AI setup: {e}"); traceback.print_exc()
        else:
            print(f"--- Opponent's turn ({opponent_player.name}) for setup ---")
            parsed_action = None
            while parsed_action is None:
                print("\nOpponent Setup Commands: active <Name>, bench <Name>, ready, pass");
                user_input = input("Enter Opponent's setup action: ").strip()
                parsed_action = parse_opponent_action(user_input, game, opponent_player)
                if parsed_action is None: print("Invalid. Try again.")
            try:
                print(f"Executing opponent setup action: {parsed_action}")
                _next_state_dict, _reward, _done = game.step(parsed_action)
            except Exception as e: print(f"Error executing opponent setup: {e}"); traceback.print_exc()

    print("\n--- Setup Phase Complete! Starting Main Game ---")
    input("Press Enter to begin Turn 1...")

 # --- Main Game Loop (MODIFIED FOR DRAW FIX) ---
    done = False
    last_ai_draw_turn = 0 # NEW: Track the last turn number AI draw was handled

    while not done:
        current_player_obj = game.game_state.get_current_player()
        is_ai_turn_main = (current_player_obj == ai_player)

        # --- Manual Draw Logic for AI (Revised Condition) ---
        # Check if it's AI's turn, not the absolute first turn, AND draw hasn't happened for this game turn yet.
        if is_ai_turn_main and not game.game_state.is_first_turn and game.game_state.turn_number > last_ai_draw_turn:
            print(f"\n--- AI Turn {game.game_state.turn_number} Draw ---") # Added turn number for clarity
            if not ai_player.deck:
                print(f"AI ({ai_player.name}) cannot draw, deck is empty!")
                # Even if deck is empty, mark this turn's draw as handled
                last_ai_draw_turn = game.game_state.turn_number
            else:
                card_added_successfully = False
                while not card_added_successfully:
                    user_card_name = input(f"Enter EXACT name of card AI ({ai_player.name}) drew: ").strip()
                    found_card_instance = None; found_index = -1
                    # Use case-insensitive search for robustness with user input
                    for i, card in enumerate(ai_player.deck):
                        if card.name.lower() == user_card_name.lower():
                            found_card_instance = card; found_index = i; break
                    if found_card_instance:
                        drawn_card = ai_player.deck.pop(found_index)
                        if len(ai_player.hand) < 10:
                            ai_player.hand.append(drawn_card)
                            print(f"--> Added '{drawn_card.name}' to AI hand.")
                        else:
                            ai_player.discard_pile.append(drawn_card)
                            print(f"--> AI hand full ({10}). Discarded '{drawn_card.name}'.")
                        card_added_successfully = True
                        # Mark draw as handled for this turn *after* successful input
                        last_ai_draw_turn = game.game_state.turn_number
                    else:
                        print(f"Error: Card '{user_card_name}' not found in AI's simulated deck.")
                        # Showing the deck can be long, maybe just show count?
                        print(f"Current AI deck ({len(ai_player.deck)} cards). Check spelling/deck list.")
                        # Keep prompting until successful

        # --- Display state AFTER potential draw ---
        display_game_state(game, ai_player)

        # --- AI Action Logic ---
        if is_ai_turn_main:
            print(f"--- AI's turn ({ai_player.name}) to make a move (Turn {game.game_state.turn_number}) ---")
            try:
                # (... Existing AI action fetching, execution, sync logic ...)
                ai_action_str = get_ai_turn_action(ai_model, vec_env, game, ai_player)
                readable_action = get_readable_ai_action(ai_action_str, ai_player)
                print(f"\nAI chooses action: {readable_action}")
                print("...")
                input("Press Enter to apply AI's action in the REAL game AND in simulation...")

                # Determine sync needs + player
                needs_sync = False; sync_card_name = None; player_to_sync = None
                trainer_prefixes = [ACTION_PLAY_SUPPORTER_PREFIX, ACTION_PLAY_ITEM_PREFIX, ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX, ACTION_PLAY_ITEM_POTION_TARGET_PREFIX, ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX, ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX]
                for prefix in trainer_prefixes:
                    if ai_action_str.startswith(prefix):
                        try:
                            action_no_prefix = ai_action_str.replace(prefix, "")
                            hand_index_str = action_no_prefix.split('_')[0]
                            hand_index = int(hand_index_str)
                            if 0 <= hand_index < len(ai_player.hand):
                                sync_card_name = ai_player.hand[hand_index].name; break
                        except (ValueError, IndexError): print(f"Warning: Could not parse hand index from '{ai_action_str}' for sync."); break

                cards_requiring_ai_sync = ["Professor's Research", "Iono", "Poké Ball"]
                cards_requiring_opp_sync = ["Mars", "Iono"]

                if sync_card_name in cards_requiring_ai_sync:
                    needs_sync = True; player_to_sync = ai_player
                if sync_card_name in cards_requiring_opp_sync:
                    needs_sync = True
                    if player_to_sync is None: player_to_sync = opponent_player

                # Execute action
                _next_state_dict, reward, step_done = game.step(ai_action_str)
                done = step_done
                print(f"Simulated action result: Reward={reward}, Game Over={done}")

                # Perform Synchronization
                if needs_sync and not done:
                    players_to_sync_list = []
                    if sync_card_name == "Iono":
                        players_to_sync_list.extend([ai_player, opponent_player])
                    elif player_to_sync:
                        players_to_sync_list.append(player_to_sync)

                    for current_sync_player in players_to_sync_list:
                        # (... Existing sync loop for each player ...)
                        sync_successful = False
                        sync_player_label = "AI" if current_sync_player == ai_player else "Opponent"
                        while not sync_successful:
                            try:
                                print(f"\nSYNC required for {sync_player_label} ({current_sync_player.name}) after '{sync_card_name}'.")
                                actual_hand_str = input(f"Enter {sync_player_label}'s ACTUAL hand (comma-separated, exact names): ")
                                actual_deck_size_str = input(f"Enter {sync_player_label}'s ACTUAL deck size: ")
                                actual_hand_names = [name.strip() for name in actual_hand_str.split(',') if name.strip()]
                                actual_deck_size = int(actual_deck_size_str)
                                if actual_deck_size < 0: print("Error: Deck size negative."); continue
                                game.synchronize_player_state(current_sync_player, actual_hand_names, actual_deck_size)
                                sync_successful = True
                                print(f"\n--- State after {sync_player_label} Synchronization ---")
                                display_game_state(game, ai_player) # Show synced state
                                print("---------------------------------")
                            except ValueError: print("Invalid input. Deck size must be number.")
                            except Exception as e: print(f"An error occurred during sync input for {sync_player_label}: {e}"); traceback.print_exc(); print("Try again.")

            except Exception as e:
                print(f"Error during AI's action: {e}")
                traceback.print_exc()
                print("Error occurred. The game will attempt to proceed.")

        # --- Opponent Action Logic ---
        else: # Opponent's turn (Human input)
            print(f"--- Opponent's turn ({opponent_player.name}) to make a move (Turn {game.game_state.turn_number}) ---")
            try:
                # (... Existing opponent action parsing, execution logic ...)
                parsed_action = None
                print("\nEnter Opponent's Action:")
                print("-------------------------")
                print("  pass | attack <idx> | energy active | energy bench <idx> | ability active | ability bench <idx> | retreat <bench_idx>")
                print("  bench <Pokemon Name> | supporter <Name> [target..] | item <Name> [target..] | tool <Name> <active|bench idx>")
                print("-------------------------")
                while parsed_action is None:
                    user_input = input("Enter Opponent's action: ").strip()
                    parsed_action = parse_opponent_action(user_input, game, opponent_player)
                    if parsed_action is None: print("Invalid input or action. Try again.")
                print(f"Executing opponent's action in simulation: {parsed_action}")
                _next_state_dict, reward, step_done = game.step(parsed_action)
                done = step_done
                print(f"Simulated action result: Reward={reward}, Game Over={done}")
            except Exception as e:
                print(f"Error executing opponent action '{parsed_action}': {e}")
                traceback.print_exc()


        # --- End of Loop Checks ---
        if game.game_state.turn_number > game.turn_limit and not done:
            print(f"Turn limit ({game.turn_limit}) reached!")
            done = True

        if not done:
            # Prompt before next loop iteration (which might be AI again or opponent)
            input("\nPress Enter to proceed to the next action or turn...")


    # --- Game Over Display ---
    # (... Game over logic remains the same ...)
    print("\n" + "="*30 + "\n--- GAME OVER ---")
    print(f"Final Turn: {game.game_state.turn_number}")
    winner = None
    if game.player1.points >= POINTS_TO_WIN: winner = game.player1
    elif game.player2.points >= POINTS_TO_WIN: winner = game.player2
    else:
        p1_lost = game.player1.active_pokemon and game.player1.active_pokemon.is_fainted and not any(p and not p.is_fainted for p in game.player1.bench)
        p2_lost = game.player2.active_pokemon and game.player2.active_pokemon.is_fainted and not any(p and not p.is_fainted for p in game.player2.bench)
        if p1_lost and not p2_lost: winner = game.player2
        elif p2_lost and not p1_lost: winner = game.player1
    if winner: print(f"Winner: {winner.name} ({'AI' if winner == ai_player else 'Opponent'})")
    elif game.game_state.turn_number > game.turn_limit: print("Result: Draw (Turn Limit Reached)")
    else:
        if game.player1.points > game.player2.points: winner = game.player1
        elif game.player2.points > game.player1.points: winner = game.player2
        if winner: print(f"Winner by points: {winner.name} ({'AI' if winner == ai_player else 'Opponent'})")
        else: print("Result: Draw (Points tied or game ended unexpectedly)")
    display_game_state(game, ai_player, show_opponent_hand=True)
    print("="*30)

def get_readable_ai_action(action_str: str, ai_player: Player) -> str:
    """Converts an internal action string into a more human-readable format."""
    try:
        parts = action_str.split('_'); prefix = ""; indices = []
        if len(parts) > 1:
            if action_str.startswith(ACTION_ATTACH_TOOL_BENCH_PREFIX):
                prefix = ACTION_ATTACH_TOOL_BENCH_PREFIX
                if len(parts) == len(prefix.split('_')) + 1: indices = [int(parts[-2]), int(parts[-1])]
                else: raise ValueError("Malformed tool bench action")
            else:
                try: indices = [int(parts[-1])]; prefix = "_".join(parts[:-1]) + "_"
                except ValueError: prefix = action_str
        else: prefix = action_str

        readable_action = action_str; card_name = None; hand_index = -1; bench_index = -1
        hand_action_prefixes = [
            ACTION_PLAY_SUPPORTER_PREFIX, ACTION_PLAY_ITEM_PREFIX, ACTION_PLAY_BASIC_BENCH_PREFIX,
            ACTION_ATTACH_TOOL_ACTIVE, ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX,
            ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX
        ]
        targeted_trainer_prefixes = [
            ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX, ACTION_PLAY_ITEM_POTION_TARGET_PREFIX,
            ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX, ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX
        ]

        if prefix in hand_action_prefixes and indices:
            hand_index = indices[0]
            if 0 <= hand_index < len(ai_player.hand): card_name = ai_player.hand[hand_index].name
        elif prefix == ACTION_ATTACH_TOOL_BENCH_PREFIX and len(indices) == 2:
            bench_index, hand_index = indices[0], indices[1]
            if 0 <= hand_index < len(ai_player.hand): card_name = ai_player.hand[hand_index].name
        elif prefix in targeted_trainer_prefixes and indices:
            try:
                hand_index_str = action_str.replace(prefix, "").split('_')[0]
                hand_index = int(hand_index_str)
                if 0 <= hand_index < len(ai_player.hand): card_name = ai_player.hand[hand_index].name
            except (ValueError, IndexError): pass

        if card_name:
            if prefix == ACTION_PLAY_SUPPORTER_PREFIX: readable_action = f"Play Supporter '{card_name}' (Hand Idx {hand_index})"
            elif prefix == ACTION_PLAY_ITEM_PREFIX: readable_action = f"Play Item '{card_name}' (Hand Idx {hand_index})"
            elif prefix == ACTION_PLAY_BASIC_BENCH_PREFIX: readable_action = f"Play Basic '{card_name}' to Bench (Hand Idx {hand_index})"
            elif prefix == ACTION_ATTACH_TOOL_ACTIVE: readable_action = f"Attach Tool '{card_name}' to Active (Hand Idx {hand_index})"
            elif prefix == ACTION_ATTACH_TOOL_BENCH_PREFIX: readable_action = f"Attach Tool '{card_name}' to Bench {bench_index} (Hand Idx {hand_index})"
            elif prefix == ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX: readable_action = f"Setup: Choose Active '{card_name}' (Hand Idx {hand_index})"
            elif prefix == ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX: readable_action = f"Setup: Choose Bench '{card_name}' (Hand Idx {hand_index})"
            elif prefix == ACTION_PLAY_ITEM_POTION_TARGET_PREFIX: readable_action = f"Play Item '{card_name}' (Potion) on Target (Hand Idx {hand_index})"
            elif prefix == ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX: readable_action = f"Play Supporter '{card_name}' (Cyrus) on Target (Hand Idx {hand_index})"
            elif prefix == ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX: readable_action = f"Play Supporter '{card_name}' (PCL) on Target (Hand Idx {hand_index})"
            elif prefix == ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX: readable_action = f"Play Supporter '{card_name}' (Dawn) from Source (Hand Idx {hand_index})"
            if action_str != readable_action: readable_action += f" [{action_str}]" # Append original for complex targets
        elif prefix == ACTION_ATTACH_ENERGY_ACTIVE: readable_action = f"Attach Energy to Active [{action_str}]"
        elif prefix == ACTION_ATTACH_ENERGY_BENCH_PREFIX and indices: readable_action = f"Attach Energy to Bench {indices[0]} [{action_str}]"
        elif prefix == ACTION_ATTACK_PREFIX and indices: readable_action = f"Attack with index {indices[0]} [{action_str}]"
        elif prefix == ACTION_USE_ABILITY_ACTIVE: readable_action = f"Use Ability on Active [{action_str}]"
        elif prefix == ACTION_USE_ABILITY_BENCH_PREFIX and indices: readable_action = f"Use Ability on Bench {indices[0]} [{action_str}]"
        elif prefix == ACTION_RETREAT_TO_BENCH_PREFIX and indices: readable_action = f"Retreat Active to Bench {indices[0]} [{action_str}]"
        elif prefix == ACTION_PASS: readable_action = "Pass Turn"
        elif prefix == ACTION_SETUP_CONFIRM_READY: readable_action = "Confirm Setup Ready"

        return readable_action
    except Exception as e:
        print(f"[Error formatting action '{action_str}': {e}]")
        return action_str

if __name__ == "__main__":
    main()