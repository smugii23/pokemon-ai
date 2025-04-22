import random
import json
import os
import copy
from typing import List, Optional, Dict, Tuple, Any
from simulator.entities import Player, GameState, PokemonCard, TrainerCard, Card, Attack, POINTS_TO_WIN

CARD_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'cards.json')
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
ACTION_USE_ABILITY_BENCH_PREFIX = "USE_ABILITY_BENCH_"

# taget specific supporters
ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX = "PLAY_SUPPORTER_CYRUS_TARGET_" 
ACTION_PLAY_ITEM_POTION_TARGET_PREFIX = "PLAY_ITEM_POTION_TARGET_" 
ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX = "PLAY_SUPPORTER_PCL_TARGET_"
ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX = "PLAY_SUPPORTER_DAWN_SOURCE_TARGET_"

ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX = "setup_choose_active_"
ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX = "setup_choose_bench_"
ACTION_SETUP_CONFIRM_READY = "setup_confirm_ready"

ACTION_OPP_PLAY_SUPPORTER_PREFIX = "OPP_PLAY_SUPPORTER_"
ACTION_OPP_PLAY_ITEM_PREFIX = "OPP_PLAY_ITEM_"
ACTION_OPP_PLAY_BASIC_PREFIX = "OPP_PLAY_BASIC_"
ACTION_OPP_ATTACH_TOOL_PREFIX = "OPP_ATTACH_TOOL_"

class Game:
    def __init__(self, player1_deck_names: List[str], player2_deck_names: List[str], player1_energy_types: List[str], player2_energy_types: List[str]):
        self.card_data = self._load_card_data()
        if not self.card_data:
            raise ValueError("Failed to load card data. Cannot initialize game.")

        # make sure the input deck lists have the correct size
        if len(player1_deck_names) != 20 or len(player2_deck_names) != 20:
            print(f"Input deck name lists should have 20 cards. P1: {len(player1_deck_names)}, P2: {len(player2_deck_names)}")

        self.player1 = Player("Player 1")
        self.player2 = Player("Player 2")
        # assign deck energy types
        self.player1.deck_energy_types = player1_energy_types if player1_energy_types else ["Colorless"]
        self.player2.deck_energy_types = player2_energy_types if player2_energy_types else ["Colorless"]

        # build the decks from the name
        deck1_cards = self._build_deck(player1_deck_names)
        deck2_cards = self._build_deck(player2_deck_names)

        # setup game for each player using the built decks
        self.player1.setup_game(deck1_cards)
        self.player2.setup_game(deck2_cards)

        # setup the game state for the players
        self.game_state = GameState(self.player1, self.player2)
        self.turn_limit = 15 # prevent infinite loops in simulation
        self.actions_this_turn: Dict[str, Any] = {} # track actions

        print(f"--- Game Start ---")
        print(f"Player 1 Energy Types: {self.player1.deck_energy_types}")
        print(f"Player 2 Energy Types: {self.player2.deck_energy_types}")
        print(f"Starting Player: {self.game_state.get_current_player().name}")
        self._initialize_energy_stand() # sets up energy according to what the player picked

    def _load_card_data(self) -> Dict[str, Dict]:
        """Loads card definitions from the JSON file."""
        try:
            with open(CARD_DATA_FILE, 'r') as f:
                all_cards_data = json.load(f)
            card_dict = {card['name']: card for card in all_cards_data}
            print(f"Successfully loaded data for {len(card_dict)} cards from {CARD_DATA_FILE}")
            return card_dict
        except FileNotFoundError:
            print(f"Error: Card data file not found at {CARD_DATA_FILE}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {CARD_DATA_FILE}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred loading card data: {e}")
            return {}

    def _build_deck(self, deck_names: List[str]) -> List[Card]:
        """Builds a deck from a list of cards in the card data"""
        deck_cards: List[Card] = []
        for name in deck_names:
            card_info = self.card_data.get(name)
            if not card_info:
                print(f"Warning: Card name '{name}' not found in loaded card data. Skipping.")
                continue

            # make sure each card in the deck is a unique instance
            card_info_copy = copy.deepcopy(card_info)
            # instantiate attacks
            attacks = []
            if card_info_copy.get("attacks"):
                for attack_data in card_info_copy["attacks"]:
                    attacks.append(Attack(
                        name=attack_data.get("name", "Unknown Attack"),
                        cost=attack_data.get("cost", {}),
                        damage=attack_data.get("damage", 0),
                        effect=attack_data.get("effect_tag")
                    ))

            # instantiate cards based on card type
            card_type = card_info_copy.get("card_type")
            
            # if it's a pokemon, get name, hp, attacks, type, weakness, retreat cost, if it's an ex and basic, and if it has an ability.
            if card_type == "Pokemon":
                pokemon = PokemonCard(
                    name=card_info_copy.get("name", "Unknown Pokemon"),
                    hp=card_info_copy.get("hp", 0),
                    attacks=attacks,
                    pokemon_type=card_info_copy.get("pokemon_type", "Colorless"),
                    weakness_type=card_info_copy.get("weakness_type"),
                    retreat_cost=card_info_copy.get("retreat_cost", 0),
                    is_ex=card_info_copy.get("is_ex", False),
                    is_basic=card_info_copy.get("is_basic", True),
                    ability=card_info_copy.get("ability")
                )
                deck_cards.append(pokemon)
            # handle all other cards (supporter, item, or tool)
            elif card_type in ["Supporter", "Item", "Tool"]:
                trainer = TrainerCard(
                    name=card_info_copy.get("name", "Unknown Trainer"),
                    trainer_type=card_type,
                    effect_tag=card_info_copy.get("effect_tag")
                )
                deck_cards.append(trainer)

        return deck_cards


    def _initialize_energy_stand(self):
        """Sets the initial state of the energy stand for both players."""
        # the player actually going first will have their available stay none after _start_turn
        # the player going second will get their preview promoted in their first _start_turn
        self.player1.energy_stand_preview = random.choice(self.player1.deck_energy_types) if self.player1.deck_energy_types else None
        self.player1.energy_stand_available = None

        self.player2.energy_stand_preview = random.choice(self.player2.deck_energy_types) if self.player2.deck_energy_types else None
        self.player2.energy_stand_available = None


    def _start_turn(self):
        """Handles start-of-turn procedures including energy stand update and drawing a card."""
        player = self.game_state.get_current_player()
        # reset turn actions as it is the beginning of a turn
        self.actions_this_turn = {
            "energy_attached": False,
            "supporter_played": False,
            "retreat_used": False,
            "red_effect_active": False,
            "leaf_effect_active": False
        }

        # promote preview to available (if available is empty/used)
        if player.energy_stand_available is None:
            player.energy_stand_available = player.energy_stand_preview
            player.energy_stand_preview = None

        # generate new preview energy randomly from deck types
        if player.deck_energy_types:
            player.energy_stand_preview = random.choice(player.deck_energy_types)
        else:
            player.energy_stand_preview = None

        print(f"{player.name} Energy Stand - Available: {player.energy_stand_available or 'None'}, Preview: {player.energy_stand_preview or 'None'}")
        # automatic draw is used for play_via_relay where the opponent's hand is unknown
        if not player.skip_automatic_draw:
            player.draw_cards(1)
        else:
            print(f"Skipping automatic draw for {player.name} (Manual draw expected in relay script).")


    def get_state_representation(self, player: Player) -> Dict[str, Any]:
        """
        Gather all relevant information about the current game state from the perspective of the
        given player, so that the AI can play/learn.
        """
        opponent = self.game_state.get_opponent()

        my_active_details = self._get_pokemon_details(player.active_pokemon)
        my_bench_details = [self._get_pokemon_details(p) for p in player.bench]

        opp_active_details = self._get_pokemon_details(opponent.active_pokemon)
        opp_bench_details = [self._get_pokemon_details(p) for p in opponent.bench]


        state_dict = {
            # player info
            "my_hand_size": len(player.hand),
            "my_hand_cards": [c.name for c in player.hand], # list the actual card names
            "my_deck_size": len(player.deck),
            "my_discard_size": len(player.discard_pile),
            "my_discard_pile_cards": [c.name for c in player.discard_pile], # knowing discard pile cards is useful for estimating probabilities for future draws
            "my_points": player.points,
            "my_energy_stand_available": player.energy_stand_available,
            "my_energy_stand_preview": player.energy_stand_preview,
            "my_active_pokemon": my_active_details,
            "my_bench_pokemon": my_bench_details,

            # opponent info
            "opp_hand_size": len(opponent.hand),
            "opp_deck_size": len(opponent.deck),
            "opp_discard_size": len(opponent.discard_pile),
            "opp_discard_pile_cards": [c.name for c in opponent.discard_pile], # opponent discard pile is useful as well
            "opp_points": opponent.points,
            "opp_energy_stand_status": {
                "available_exists": opponent.energy_stand_available is not None,
                "preview": opponent.energy_stand_preview
            },
            "opp_active_pokemon": opp_active_details,
            "opp_bench_pokemon": opp_bench_details,

            # info about the game
            "turn": self.game_state.turn_number,
            "is_my_turn": player == self.game_state.get_current_player(),
            "can_attach_energy": not self.actions_this_turn.get("energy_attached", False),
            "is_first_turn": self.game_state.is_first_turn,
        }
        return state_dict

    def _get_pokemon_details(self, pokemon: Optional[PokemonCard]) -> Optional[Dict[str, Any]]: # Return type is Optional[Dict]
        """
        Extract relevant details from a PokemonCard object for state representation.
        Returns None if pokemon is None or fainted.
        """
        if pokemon is None or pokemon.is_fainted:
            return None

        details = {
            "name": pokemon.name,
            "hp": pokemon.hp,
            "current_hp": pokemon.current_hp,
            "attached_energy": pokemon.attached_energy.copy(),
            "attack_names": [attack.name for attack in pokemon.attacks],
            "attached_tool_name": pokemon.attached_tool.name if pokemon.attached_tool else None,
            "pokemon_type": pokemon.pokemon_type,
            "weakness_type": pokemon.weakness_type,
            "is_ex": pokemon.is_ex
        }
        return details

    def get_possible_actions(self) -> List[str]:
        """
        Get a list of valid actions for the current player.
        """
        player = self.game_state.get_current_player()
        opponent = self.game_state.get_opponent()
        actions = []
        player_idx = self.game_state.current_player_index
        opponent_idx = 1 - player_idx

        # setup phase where both players place down a pokemon onto the board (must place a pokemon in the active slot) and cannot see each others pokemon
        # until they both select ready
        is_setup_phase_check = not player.setup_ready or not opponent.setup_ready
        if is_setup_phase_check:
            # check if player has set up
            if not player.setup_ready:
                # check for pokemon in active slot
                if player.pending_setup_active is None:
                    found_basic = False
                    for i, card in enumerate(player.hand):
                        if isinstance(card, PokemonCard) and card.is_basic:
                            action_name = f"{ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX}{i}"
                            actions.append(action_name)
                            found_basic = True
                    if not found_basic:
                        print(f"CRITICAL ERROR: Player {player.name} has no basic Pokemon in hand during setup choice!")
                        return []
                    return actions

                # 2. can choose bench pokemon (optional) if active is chosen and bench isn't full
                # need to track which hand indices are already assigned to pending active/bench
                assigned_hand_indices = set()
                if player.pending_setup_active:
                    # Find the hand index corresponding to the pending active card
                    # This assumes card instances are unique or we track index on selection
                    try:
                        active_card_instance = player.pending_setup_active
                        active_index = player.hand.index(active_card_instance)
                        assigned_hand_indices.add(active_index)
                    except ValueError:
                         print(f"Warning: Could not find pending active card {player.pending_setup_active.name} in hand during action generation.")

                for bench_card_instance in player.pending_setup_bench:
                    try:
                        bench_index = player.hand.index(bench_card_instance)
                        assigned_hand_indices.add(bench_index)
                    except ValueError:
                         print(f"Warning: Could not find pending bench card {bench_card_instance.name} in hand during action generation.")


                can_add_to_bench = len(player.pending_setup_bench) < MAX_BENCH_SIZE
                if can_add_to_bench:
                    for i, card in enumerate(player.hand):
                        # Check if card is basic and not already assigned to active/bench
                        if i not in assigned_hand_indices and isinstance(card, PokemonCard) and card.is_basic:
                            action_name = f"{ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX}{i}"
                            actions.append(action_name)

                # 3. Can confirm ready if Active has been chosen
                if player.pending_setup_active is not None:
                    actions.append(ACTION_SETUP_CONFIRM_READY)
                return actions # Return only setup actions for the player who isn't ready

            # If the current player *is* ready, but the opponent is not, player must PASS (wait)
            else: # player.setup_ready is True, but opponent.setup_ready is False
                return [ACTION_PASS] # Only action is to wait

        # check the possible actions:
        can_attach_energy_this_turn = not self.actions_this_turn.get("energy_attached", False)
        # The absolute first turn of the game cannot attach energy or attack.
        is_game_first_turn = self.game_state.is_first_turn

        # attach energy
        if player.energy_stand_available and can_attach_energy_this_turn and not is_game_first_turn:
             # print(f"DEBUG: Condition met for attaching energy.") # DEBUG
             # check potential targets (active and bench)
             if player.active_pokemon and not player.active_pokemon.is_fainted:
                 # print(f"DEBUG: Adding action: {ACTION_ATTACH_ENERGY_ACTIVE}") # DEBUG
                 actions.append(ACTION_ATTACH_ENERGY_ACTIVE)
             for i, bench_pokemon in enumerate(player.bench):
                 if bench_pokemon and not bench_pokemon.is_fainted:
                     action_name = f"{ACTION_ATTACH_ENERGY_BENCH_PREFIX}{i}"
                     # print(f"DEBUG: Adding action: {action_name}") # DEBUG
                     actions.append(action_name)
        # else: # DEBUG
            # print(f"DEBUG: Condition NOT met for attaching energy.") # DEBUG
            # if not player.energy_stand_available: print("  Reason: No energy available in stand.") # DEBUG
            # if not can_attach_energy_this_turn: print("  Reason: Energy already attached this turn.") # DEBUG
            # if is_game_first_turn: print("  Reason: Is the absolute first turn of the game.") # DEBUG


        # check attack actions (cannot attack on first turn)
        if player.active_pokemon and not player.active_pokemon.is_fainted and not is_game_first_turn:
            # print(f"DEBUG: Checking attacks for {player.active_pokemon.name} (Energy: {player.active_pokemon.attached_energy})") # DEBUG
            for i, attack in enumerate(player.active_pokemon.attacks):
                 # print(f"DEBUG:  Checking attack '{attack.name}' (Cost: {attack.cost})") # DEBUG
                 can_attack_flag = player.active_pokemon.can_attack(i)
                 # print(f"DEBUG:  can_attack({i}) returned: {can_attack_flag}") # DEBUG
                 if can_attack_flag:
                     action_name = f"{ACTION_ATTACK_PREFIX}{i}"
                     # print(f"DEBUG: Adding action: {action_name}") # DEBUG
                     actions.append(action_name)
        # else: # DEBUG
            # if not player.active_pokemon: print("DEBUG: No active pokemon to attack with.") # DEBUG
            # elif player.active_pokemon.is_fainted: print("DEBUG: Active pokemon is fainted.") # DEBUG
            # elif is_game_first_turn: print("DEBUG: Cannot attack on the first turn.") # DEBUG


        # play a basic pokemon onto the bench
        if player.can_place_on_bench():
            # print(f"DEBUG: Checking hand for basic Pokemon to bench.") # DEBUG
            for i, card in enumerate(player.hand):
                 if isinstance(card, PokemonCard) and card.is_basic:
                     action_name = f"{ACTION_PLAY_BASIC_BENCH_PREFIX}{i}"
                     # print(f"DEBUG: Adding action: {action_name} (Card: {card.name})") # DEBUG
                     actions.append(action_name)
        # else: # DEBUG
            # print("DEBUG: Cannot place on bench (bench full).") # DEBUG

        # Check for using ACTIVE ability
        if player.active_pokemon and player.active_pokemon.ability:
            ability_data = player.active_pokemon.ability
            ability_name = ability_data.get("name")
            ability_type = ability_data.get("type")
            ability_used_key = f"ability_used_{ability_name}"
            # print(f"DEBUG: Checking ability '{ability_name}' (Type: {ability_type}, Used: {self.actions_this_turn.get(ability_used_key, False)})") # DEBUG
            # Allow any Active ability that hasn't been used this turn
            if ability_type == "Active" and not self.actions_this_turn.get(ability_used_key, False):
                 # print(f"DEBUG: Adding action: {ACTION_USE_ABILITY_ACTIVE}") # DEBUG
                 actions.append(ACTION_USE_ABILITY_ACTIVE)
        # Check for using BENCH abilities
        # ACTION_USE_ABILITY_BENCH_PREFIX = "USE_ABILITY_BENCH_" # Defined above
        for i, bench_pokemon in enumerate(player.bench):
             if bench_pokemon and bench_pokemon.ability:
                 ability_data = bench_pokemon.ability
                 ability_name = ability_data.get("name")
                 ability_type = ability_data.get("type")
                 ability_used_key = f"ability_used_{ability_name}" # Assume ability names are unique for now
                 # print(f"DEBUG: Checking bench {i} ability '{ability_name}' (Type: {ability_type}, Used: {self.actions_this_turn.get(ability_used_key, False)})") # DEBUG
                 # Allow any Active ability that hasn't been used this turn
                 if ability_type == "Active" and not self.actions_this_turn.get(ability_used_key, False):
                     action_name = f"{ACTION_USE_ABILITY_BENCH_PREFIX}{i}"
                     # print(f"DEBUG: Adding action: {action_name}") # DEBUG
                     actions.append(action_name)


        # Check for playing Trainer cards
        can_play_supporter = not self.actions_this_turn.get("supporter_played", False)
        # print(f"DEBUG: Checking hand for Trainers (Can play supporter: {can_play_supporter})") # DEBUG
        for i, card in enumerate(player.hand):
            if isinstance(card, TrainerCard):
                # print(f"DEBUG:  Found Trainer: {card.name} (Type: {card.trainer_type}, Effect: {card.effect_tag}) at index {i}") # DEBUG
                if card.trainer_type == "Supporter":
                    if can_play_supporter:
                        # --- Add specific checks for Supporters ---
                        can_play_this_supporter = True # Assume true initially
                        effect_tag = card.effect_tag

                        if effect_tag == "TRAINER_SUPPORTER_SABRINA_SWITCH_OUT_YOUR_OPPONENT":
                            # Check if opponent has a valid bench Pokemon to switch to
                            has_valid_bench = any(p and not p.is_fainted for p in opponent.bench)
                            if not opponent.active_pokemon or not has_valid_bench:
                                can_play_this_supporter = False
                                # print(f"DEBUG: Cannot add Sabrina action: Opponent has no valid active/bench.") # DEBUG

                        elif effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
                            # Check if opponent has a damaged, non-fainted bench Pokemon
                            has_damaged_bench = any(p and not p.is_fainted and p.current_hp < p.hp for p in opponent.bench)
                            if not opponent.active_pokemon or not has_damaged_bench:
                                can_play_this_supporter = False
                                # print(f"DEBUG: Cannot add Cyrus action: Opponent has no damaged bench.") # DEBUG

                        elif effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
                            # Check if player has a benched Pokemon with energy AND an active Pokemon
                            has_benched_energy = any(p and not p.is_fainted and p.attached_energy for p in player.bench)
                            if not player.active_pokemon or player.active_pokemon.is_fainted or not has_benched_energy:
                                can_play_this_supporter = False
                                # print(f"DEBUG: Cannot add Dawn action: No valid source/target.") # DEBUG

                        elif effect_tag == "TRAINER_SUPPORTER_TEAM_ROCKET_GRUNT_FLIP_A_COIN_UN":
                            # Check if opponent's active Pokemon exists, is not fainted, and has energy
                            if not opponent.active_pokemon or opponent.active_pokemon.is_fainted or not opponent.active_pokemon.attached_energy:
                                can_play_this_supporter = False
                                # print(f"DEBUG: Cannot add TR Grunt action: Opponent active has no energy.") # DEBUG

                        elif effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
                            # Check if any player Pokemon is damaged
                            is_any_pokemon_damaged = False
                            if player.active_pokemon and player.active_pokemon.current_hp < player.active_pokemon.hp:
                                is_any_pokemon_damaged = True
                            if not is_any_pokemon_damaged:
                                for p in player.bench:
                                    if p and p.current_hp < p.hp:
                                        is_any_pokemon_damaged = True
                                        break
                            if not is_any_pokemon_damaged:
                                can_play_this_supporter = False
                                # print(f"DEBUG: Cannot add PCL action: No damaged Pokemon.") # DEBUG

                        # --- Generate Target-Specific or Generic Supporter Actions ---
                        if effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
                            if can_play_this_supporter: # Basic check passed
                                # Find specific targets
                                for bench_idx, p in enumerate(opponent.bench):
                                    if p and not p.is_fainted and p.current_hp < p.hp:
                                        action_name = f"{ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX}{i}_{bench_idx}"
                                        actions.append(action_name)
                        elif effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
                             if can_play_this_supporter: # Basic check passed (includes damage check now)
                                 # Find specific targets
                                 if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.current_hp < player.active_pokemon.hp:
                                     action_name = f"{ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX}{i}_active"
                                     actions.append(action_name)
                                 for bench_idx, p in enumerate(player.bench):
                                     if p and not p.is_fainted and p.current_hp < p.hp:
                                         action_name = f"{ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX}{i}_bench_{bench_idx}"
                                         actions.append(action_name)
                        elif effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
                             if can_play_this_supporter: # Basic check passed
                                 # Find specific sources (target is always active)
                                 if player.active_pokemon and not player.active_pokemon.is_fainted:
                                     for bench_idx, p in enumerate(player.bench):
                                         if p and not p.is_fainted and p.attached_energy:
                                             action_name = f"{ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX}{i}_{bench_idx}"
                                             actions.append(action_name)
                        # --- Add Generic Action for other Supporters ---
                        elif can_play_this_supporter: # For supporters not handled above
                            action_name = f"{ACTION_PLAY_SUPPORTER_PREFIX}{i}"
                            actions.append(action_name)
                        # else: # DEBUG
                            # print(f"DEBUG: Supporter {card.name} condition not met.") # DEBUG

                elif card.trainer_type == "Item":
                    can_play_this_item = True # Assume true initially
                    effect_tag = card.effect_tag

                    # --- Add specific checks and Generate Target-Specific or Generic Item Actions ---
                    if effect_tag == "TRAINER_ITEM_POTION_HEAL_20_DAMAGE_FROM_1_OF_YOUR_":
                        # Check if any player Pokemon is damaged (redundant check kept for safety)
                        is_any_pokemon_damaged = False
                        if player.active_pokemon and player.active_pokemon.current_hp < player.active_pokemon.hp:
                            is_any_pokemon_damaged = True
                        if not is_any_pokemon_damaged:
                            for p in player.bench:
                                if p and p.current_hp < p.hp:
                                    is_any_pokemon_damaged = True
                                    break
                        if not is_any_pokemon_damaged:
                            can_play_this_item = False
                            # print(f"DEBUG: Cannot add Potion action: No damaged Pokemon.") # DEBUG
                        else:
                            # Generate target-specific actions
                            if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.current_hp < player.active_pokemon.hp:
                                action_name = f"{ACTION_PLAY_ITEM_POTION_TARGET_PREFIX}{i}_active"
                                actions.append(action_name)
                            for bench_idx, p in enumerate(player.bench):
                                if p and not p.is_fainted and p.current_hp < p.hp:
                                    action_name = f"{ACTION_PLAY_ITEM_POTION_TARGET_PREFIX}{i}_bench_{bench_idx}"
                                    actions.append(action_name)

                    # Add other item checks here...

                    # --- Add Generic Action for other Items ---
                    elif can_play_this_item: # For items not handled above
                        action_name = f"{ACTION_PLAY_ITEM_PREFIX}{i}"
                        actions.append(action_name)
                    # else: # DEBUG
                        # print(f"DEBUG: Item {card.name} condition not met.") # DEBUG

                elif card.trainer_type == "Tool":
                    # Check potential targets
                    if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.attached_tool is None:
                        action_name = f"{ACTION_ATTACH_TOOL_ACTIVE}{i}"
                        # print(f"DEBUG: Adding action: {action_name} (Target: Active)") # DEBUG
                        actions.append(action_name)
                    for bench_idx, bench_pokemon in enumerate(player.bench):
                        if bench_pokemon and not bench_pokemon.is_fainted and bench_pokemon.attached_tool is None:
                            action_name = f"{ACTION_ATTACH_TOOL_BENCH_PREFIX}{bench_idx}_{i}"
                            # print(f"DEBUG: Adding action: {action_name} (Target: Bench {bench_idx})") # DEBUG
                            actions.append(action_name)

        # Check for retreating
        can_retreat_this_turn = not self.actions_this_turn.get("retreat_used", False)
        if can_retreat_this_turn and player.active_pokemon and not player.active_pokemon.is_fainted and player.bench:
            # Check if there's at least one valid Pokemon to switch into on the bench
            valid_bench_targets = [i for i, p in enumerate(player.bench) if p and not p.is_fainted]
            if valid_bench_targets:
                # Check if active Pokemon can afford the retreat cost (considering Leaf)
                retreat_cost_modifier = -2 if self.actions_this_turn.get("leaf_effect_active", False) else 0
                if player.active_pokemon.can_retreat(current_turn_cost_modifier=retreat_cost_modifier):
                    # print(f"DEBUG: Checking retreat action (Can afford cost).") # DEBUG
                    for bench_idx in valid_bench_targets:
                        action_name = f"{ACTION_RETREAT_TO_BENCH_PREFIX}{bench_idx}"
                        # print(f"DEBUG: Adding action: {action_name}") # DEBUG
                        actions.append(action_name)
                # else: # DEBUG
                    # print(f"DEBUG: Cannot retreat (cannot afford cost: {player.active_pokemon.retreat_cost + retreat_cost_modifier}, Attached: {sum(player.active_pokemon.attached_energy.values())}).") # DEBUG
            # else: # DEBUG
                # print("DEBUG: Cannot retreat (no valid bench Pokemon to promote).") # DEBUG
        # else: # DEBUG
            # print(f"DEBUG: Condition NOT met for retreating.") # DEBUG
            # if not can_retreat_this_turn: print("  Reason: Already retreated this turn.") # DEBUG
            # if not player.active_pokemon: print("  Reason: No active Pokemon.") # DEBUG
            # if player.active_pokemon and player.active_pokemon.is_fainted: print("  Reason: Active Pokemon is fainted.") # DEBUG
            # if not player.bench: print("  Reason: Bench is empty.") # DEBUG


        # TODO: Add actions for evolving etc.

        # also have the option to just pass
        # print(f"DEBUG: Adding action: {ACTION_PASS}") # DEBUG
        actions.append(ACTION_PASS)

        # print(f"DEBUG: Final generated actions: {actions}") # DEBUG
        return actions

    def step(self, action: str) -> Tuple[Any, float, bool]:
        """
        Execute an action, update the game state, and return the new state, reward, and done flag.
        """
        player = self.game_state.get_current_player()
        opponent = self.game_state.get_opponent()
        reward = 0.0 # this is the default reward for taking a step
        done = False
        info = {} # just for debugging or extra info
        # ACTION_USE_ABILITY_BENCH_PREFIX = "USE_ABILITY_BENCH_" # Defined globally now

        print(f"Player {player.name} attempts action: {action}")

        action_executed = False
        is_game_first_turn_overall = self.game_state.is_first_turn # Check if it's the absolute first turn
        player_idx = self.game_state.current_player_index
        is_setup_phase = not player.setup_ready or not opponent.setup_ready

        # --- Handle NEW Simultaneous Setup Actions ---
        if is_setup_phase:
            # Actions for the player who is NOT ready yet
            if not player.setup_ready:
                if action.startswith(ACTION_SETUP_CHOOSE_ACTIVE_FROM_HAND_PREFIX):
                    if player.pending_setup_active is not None:
                        print(f"Error: Player {player.name} tried {action}, but already chose a pending active.")
                        reward = -0.5 # Increased penalty
                    else:
                        try:
                            hand_index = int(action.split('_')[-1])
                            if 0 <= hand_index < len(player.hand):
                                card_to_choose = player.hand[hand_index]
                                # Ensure card hasn't been chosen for bench already
                                already_chosen_for_bench = card_to_choose in player.pending_setup_bench
                                if isinstance(card_to_choose, PokemonCard) and card_to_choose.is_basic and not already_chosen_for_bench:
                                    player.pending_setup_active = card_to_choose # Store reference
                                    print(f"{player.name} chose {card_to_choose.name} as pending Active.")
                                    action_executed = True
                                    reward += 0.01
                                elif already_chosen_for_bench:
                                    print(f"Error: Card at index {hand_index} ({card_to_choose.name}) is already chosen for bench.")
                                    reward -= 0.5 # Increased penalty
                                else:
                                    print(f"Error: Card at index {hand_index} is not a Basic Pokemon.")
                                    reward -= 0.5 # Increased penalty
                            else:
                                print(f"Error: Invalid hand index {hand_index} in action {action}.")
                                reward -= 0.5 # Increased penalty
                        except (ValueError, IndexError):
                            print(f"Error: Invalid action format {action}.")
                            reward -= 0.5 # Increased penalty

                elif action.startswith(ACTION_SETUP_CHOOSE_BENCH_FROM_HAND_PREFIX):
                    if player.pending_setup_active is None:
                        print(f"Error: Player {player.name} tried {action}, but must choose active first.")
                        reward = -0.5 # Increased penalty
                    elif len(player.pending_setup_bench) >= MAX_BENCH_SIZE:
                        print(f"Error: Player {player.name} tried {action}, but pending bench is full.")
                        reward = -0.5 # Increased penalty
                    else:
                        try:
                            hand_index = int(action.split('_')[-1])
                            if 0 <= hand_index < len(player.hand):
                                card_to_choose = player.hand[hand_index]
                                # Check if card is already chosen for active or bench
                                already_chosen = (card_to_choose == player.pending_setup_active or
                                                  card_to_choose in player.pending_setup_bench)

                                if isinstance(card_to_choose, PokemonCard) and card_to_choose.is_basic and not already_chosen:
                                    player.pending_setup_bench.append(card_to_choose) # Store reference
                                    print(f"{player.name} added {card_to_choose.name} to pending Bench.")
                                    action_executed = True
                                    reward += 0.01
                                elif already_chosen:
                                    print(f"Error: Card at index {hand_index} ({card_to_choose.name}) is already chosen for setup.")
                                    reward -= 0.5 # Increased penalty
                                else:
                                    print(f"Error: Card at index {hand_index} is not a Basic Pokemon.")
                                    reward -= 0.5 # Increased penalty
                            else:
                                print(f"Error: Invalid hand index {hand_index} in action {action}.")
                                reward -= 0.5 # Increased penalty
                        except (ValueError, IndexError):
                            print(f"Error: Invalid action format {action}.")
                            reward -= 0.5 # Increased penalty

                elif action == ACTION_SETUP_CONFIRM_READY:
                    if player.pending_setup_active is None:
                        print(f"Error: Player {player.name} tried to confirm ready without choosing an active.")
                        reward = -0.5 # Increased penalty
                    else:
                        player.setup_ready = True
                        action_executed = True
                        reward += 0.01
                        print(f"{player.name} confirmed setup readiness.")
                        # Check if opponent is also ready
                        if opponent.setup_ready:
                            print("Both players ready! Committing setup choices.")
                            self._commit_setup_choices()
                            # Setup complete, start the first turn for the designated starting player
                            print("Setup complete. Starting first turn.")
                            self._start_turn() # This sets the correct player and turn state
                            # Game now proceeds to normal turn logic for the starting player
                        else:
                            # Opponent not ready, manually switch index so they can act, DO NOT call switch_turn()
                            print(f"{player.name} is ready, switching to {opponent.name} for setup (manual index flip).")
                            self.game_state.current_player_index = 1 - self.game_state.current_player_index
                            # self.game_state.switch_turn() # DO NOT CALL THIS HERE

                elif action == ACTION_PASS:
                     print(f"Error: Player {player.name} tried to pass while still needing to choose setup.")
                     reward = -0.5 # Increased penalty
                     action_executed = False # Invalid pass
                else:
                     reward = -0.5 # Increased penalty
                     action_executed = False

            # Action for the player who IS ready, but opponent is not
            elif player.setup_ready and not opponent.setup_ready:
                if action == ACTION_PASS:
                    print(f"{player.name} is ready and passes, waiting for {opponent.name} (manual index flip).")
                    action_executed = True
                    # Manually switch index so opponent can continue setup, DO NOT call switch_turn()
                    self.game_state.current_player_index = 1 - self.game_state.current_player_index
                    # self.game_state.switch_turn() # DO NOT CALL THIS HERE
                else:
                    print(f"Error: Player {player.name} is ready but opponent is not. Only PASS is allowed. Tried '{action}'.")
                    reward = -0.5 # Increased penalty
                    action_executed = False

        # --- Handle Normal Turn Actions (Only if setup is complete) ---
        # --- CORRECTED if/elif structure for normal actions ---
        elif not is_setup_phase:
            action_was_known = True # Flag to track if action matched a known pattern
            if action == ACTION_PASS:
                print("Turn passed.")
                action_executed = True # Passing is always a valid execution

            elif action == ACTION_ATTACH_ENERGY_ACTIVE:
                # Ensure player has an active pokemon first (should be true after setup)
                if not player.active_pokemon:
                     print("Error: Cannot attach energy, player has no active Pokemon (Setup likely incomplete).")
                     reward = -0.5 # Increased penalty
                     # action_executed remains False
                elif is_game_first_turn_overall: # Use the flag checked at the start of step
                     print("Error: Cannot attach energy on the first turn of the game.")
                     reward = -0.5 # Increased penalty
                     # action_executed remains False
                else:
                    can_attach_energy_this_turn = not self.actions_this_turn.get("energy_attached", False)
                    energy_to_attach = player.energy_stand_available # get the energy type from the stand

                    if energy_to_attach and can_attach_energy_this_turn and player.active_pokemon:
                          # Check energy count BEFORE attaching
                          current_energy_count = sum(player.active_pokemon.attached_energy.values())
                          reward += 0.03
                          player.active_pokemon.attach_energy(energy_to_attach)
                          player.energy_stand_available = None # actually consume the energy form the energy stand
                          self.actions_this_turn["energy_attached"] = True
                          action_executed = True
                          # Reward is now set above based on condition

                          # --- Check for Nightmare Aura (Darkrai ex) ---
                          if player.active_pokemon.name == "Darkrai ex" and energy_to_attach == "Darkness":
                             print(f"Ability Trigger: {player.active_pokemon.name}'s Nightmare Aura!")
                             opponent = self.game_state.get_opponent()
                             if opponent.active_pokemon and not opponent.active_pokemon.is_fainted:
                                 opponent.active_pokemon.take_damage(20)
                                 print(f"  Nightmare Aura dealt 20 damage to {opponent.active_pokemon.name}.")
                                 # Check for KO from ability damage
                                 if opponent.active_pokemon.is_fainted:
                                     points_scored = 2 if opponent.active_pokemon.is_ex else 1
                                     print(f"  {opponent.active_pokemon.name} fainted from Nightmare Aura. {player.name} scores {points_scored} point(s).")
                                     player.add_point(points_scored)
                                     reward += 0.75 * points_scored
                                     winner = self.game_state.check_win_condition()
                                     if winner:
                                         print(f"Game Over. Winner: {winner.name}")
                                         reward += 2.0 # Big reward for winning
                                         done = True
                                     elif not opponent.promote_bench_pokemon():
                                          print(f"Game Over. {opponent.name} has no Pokemon to promote. Winner: {player.name}")
                                          reward += 2.0 # Reward for winning
                                          done = True

                    else: # This else corresponds to the 'if energy_to_attach and can_attach_energy_this_turn and player.active_pokemon:'
                         print(f"Cannot attach energy (Available: {energy_to_attach}, Attached This Turn: {not can_attach_energy_this_turn}).")
                         # Keep penalty for trying illegal attach
                         if energy_to_attach and not can_attach_energy_this_turn:
                             reward -= 0.5 # Increased penalty
                         # action_executed remains False
            elif action.startswith(ACTION_OPP_PLAY_SUPPORTER_PREFIX):
                can_play_supporter = not self.actions_this_turn.get("supporter_played", False)
                if not can_play_supporter:
                    print("Error: Cannot play Supporter: Already played one this turn.")
                    reward -= 0.5
                    action_executed = False # Invalid action timing
                else:
                    try:
                        # Extract Card Name and Target Info
                        # Format: OPP_PLAY_SUPPORTER_Card-Name-With-Hyphens[_TargetInfo...]
                        action_parts = action.replace(ACTION_OPP_PLAY_SUPPORTER_PREFIX, "").split('_')
                        card_name_hyphenated = action_parts[0]
                        card_name = card_name_hyphenated.replace('-', ' ') # Convert back
                        target_info_parts = action_parts[1:]

                        ghost_card = self._instantiate_card_from_name(card_name)

                        if isinstance(ghost_card, TrainerCard) and ghost_card.trainer_type == "Supporter":
                            print(f"Opponent plays Supporter (Ghost): {ghost_card.name}")

                            # --- Parse Target Info based on card ---
                            target_kwargs = {} # Arguments for _execute_trainer_effect
                            effect_tag = ghost_card.effect_tag
                            parse_error = False
                            if effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
                                if len(target_info_parts) == 1:
                                     try: target_kwargs['target_bench_index'] = int(target_info_parts[0])
                                     except ValueError: parse_error = True
                                else: parse_error = True
                                if parse_error: print(f"Error: Invalid target format for Cyrus ghost card. Expected bench index.")
                            elif effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
                                if len(target_info_parts) == 1 and target_info_parts[0] == "active":
                                    target_kwargs['target_id'] = "active"
                                elif len(target_info_parts) == 2 and target_info_parts[0] == "bench":
                                    try: target_kwargs['target_id'] = f"bench_{int(target_info_parts[1])}"
                                    except ValueError: parse_error = True
                                else: parse_error = True
                                if parse_error: print(f"Error: Invalid target format for PCL ghost card. Expected 'active' or 'bench <idx>'.")
                            elif effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
                                if len(target_info_parts) == 1:
                                     try: target_kwargs['source_bench_index'] = int(target_info_parts[0])
                                     except ValueError: parse_error = True
                                else: parse_error = True
                                if parse_error: print(f"Error: Invalid target format for Dawn ghost card. Expected source bench index.")
                            # Add more target parsing here if needed for other opponent supporters

                            if parse_error:
                                 reward -= 0.5
                                 action_executed = False
                            else:
                                action_executed = True # Assume valid play attempt
                                self.actions_this_turn["supporter_played"] = True
                                effect_executed_successfully = self._execute_trainer_effect(
                                    effect_tag, player, opponent, **target_kwargs # Pass parsed targets
                                )
                                # Add ghost card to discard AFTER effect attempt
                                player.discard_pile.append(ghost_card)
                                if effect_executed_successfully:
                                    reward += 0.05
                                else:
                                    print(f"Effect of Opponent's {ghost_card.name} failed or had no target.")
                                    # Don't penalize reward if action was valid but effect fizzled
                        elif ghost_card is None:
                            print(f"Error: Could not instantiate opponent supporter '{card_name}'.")
                            reward -= 0.5
                            action_executed = False
                        else:
                            print(f"Error: Opponent tried to play non-Supporter card '{card_name}' as supporter.")
                            reward -= 0.5
                            action_executed = False
                    except Exception as e:
                        print(f"Error parsing opponent supporter action '{action}': {e}")
                        reward -= 0.5
                        action_executed = False

            # --- Handle Opponent Play Item (Ghost Card) ---
            elif action.startswith(ACTION_OPP_PLAY_ITEM_PREFIX):
                 try:
                     # Format: OPP_PLAY_ITEM_Card-Name-With-Hyphens[_TargetInfo...]
                     action_parts = action.replace(ACTION_OPP_PLAY_ITEM_PREFIX, "").split('_')
                     card_name_hyphenated = action_parts[0]
                     card_name = card_name_hyphenated.replace('-', ' ')
                     target_info_parts = action_parts[1:]

                     ghost_card = self._instantiate_card_from_name(card_name)

                     if isinstance(ghost_card, TrainerCard) and ghost_card.trainer_type == "Item":
                         print(f"Opponent plays Item (Ghost): {ghost_card.name}")

                         # --- Parse Target Info based on card ---
                         target_kwargs = {}
                         effect_tag = ghost_card.effect_tag
                         parse_error = False
                         if effect_tag == "TRAINER_ITEM_POTION_HEAL_20_DAMAGE_FROM_1_OF_YOUR_":
                             if len(target_info_parts) == 1 and target_info_parts[0] == "active":
                                 target_kwargs['target_id'] = "active"
                             elif len(target_info_parts) == 2 and target_info_parts[0] == "bench":
                                 try: target_kwargs['target_id'] = f"bench_{int(target_info_parts[1])}"
                                 except ValueError: parse_error = True
                             else: parse_error = True
                             if parse_error: print(f"Error: Invalid target format for Potion ghost card. Expected 'active' or 'bench <idx>'.")
                         # Add more target parsing here if needed for other opponent items

                         if parse_error:
                              reward -= 0.5
                              action_executed = False
                         else:
                             action_executed = True # Assume valid play attempt
                             effect_executed_successfully = self._execute_trainer_effect(
                                 effect_tag, player, opponent, **target_kwargs
                             )
                             # Add ghost card to discard AFTER effect attempt
                             player.discard_pile.append(ghost_card)
                             if effect_executed_successfully:
                                 reward += 0.05
                             else:
                                 print(f"Effect of Opponent's {ghost_card.name} failed or had no target.")
                     elif ghost_card is None:
                         print(f"Error: Could not instantiate opponent item '{card_name}'.")
                         reward -= 0.5
                         action_executed = False
                     else:
                         print(f"Error: Opponent tried to play non-Item card '{card_name}' as item.")
                         reward -= 0.5
                         action_executed = False
                 except Exception as e:
                     print(f"Error parsing opponent item action '{action}': {e}")
                     reward -= 0.5
                     action_executed = False

            # --- Handle Opponent Play Basic Pokemon (Ghost Card) ---
            elif action.startswith(ACTION_OPP_PLAY_BASIC_PREFIX):
                if player.can_place_on_bench():
                    try:
                        # Format: OPP_PLAY_BASIC_Pokemon-Name-With-Hyphens
                        card_name_hyphenated = action.replace(ACTION_OPP_PLAY_BASIC_PREFIX, "")
                        card_name = card_name_hyphenated.replace('-', ' ')

                        ghost_card = self._instantiate_card_from_name(card_name)

                        if isinstance(ghost_card, PokemonCard) and ghost_card.is_basic:
                            # Place the instantiated ghost card directly onto the bench
                            player.place_on_bench(ghost_card) # This function handles printing
                            action_executed = True
                            reward += 0.01
                        elif ghost_card is None:
                             print(f"Error: Could not instantiate opponent basic Pokemon '{card_name}'.")
                             reward -= 0.5
                             action_executed = False
                        else:
                            print(f"Error: Opponent tried to play non-Basic Pokemon card '{card_name}' to bench.")
                            reward -= 0.5
                            action_executed = False
                    except Exception as e:
                         print(f"Error parsing opponent play basic action '{action}': {e}")
                         reward -= 0.5
                         action_executed = False
                else:
                     print("Opponent cannot play to bench: Bench is full.")
                     reward -= 0.5 # Penalize if user tried this when bench full
                     action_executed = False # Action failed

            # --- Handle Opponent Attach Tool (Ghost Card) ---
            elif action.startswith(ACTION_OPP_ATTACH_TOOL_PREFIX):
                 try:
                     # Format: OPP_ATTACH_TOOL_Tool-Name-With-Hyphens_TargetType[_TargetIndex]
                     action_parts = action.replace(ACTION_OPP_ATTACH_TOOL_PREFIX, "").split('_')
                     card_name_hyphenated = action_parts[0]
                     card_name = card_name_hyphenated.replace('-', ' ')
                     target_type = action_parts[1] # Should be 'active' or 'bench'
                     target_index_str = action_parts[2] if len(action_parts) > 2 else None

                     ghost_card = self._instantiate_card_from_name(card_name)
                     target_pokemon: Optional[PokemonCard] = None

                     if isinstance(ghost_card, TrainerCard) and ghost_card.trainer_type == "Tool":
                         # Find target Pokemon
                         if target_type == "active":
                             target_pokemon = player.active_pokemon
                         elif target_type == "bench" and target_index_str is not None:
                             try:
                                 bench_index = int(target_index_str)
                                 if 0 <= bench_index < len(player.bench):
                                     target_pokemon = player.bench[bench_index]
                                 else:
                                     print(f"Error: Invalid bench index '{bench_index}' for opponent tool attach.")
                             except ValueError:
                                 print(f"Error: Invalid bench index format '{target_index_str}'.")
                         else:
                             print(f"Error: Invalid target type '{target_type}' for opponent tool attach.")

                         # Attach if valid target found and can attach
                         if target_pokemon and not target_pokemon.is_fainted and target_pokemon.attached_tool is None:
                             target_pokemon.attached_tool = ghost_card
                             print(f"Opponent attached {ghost_card.name} to {target_pokemon.name}.")
                             # Apply immediate effects (e.g., HP boost)
                             if ghost_card.effect_tag == "TRAINER_TOOL_GIANT_CAPE_THE_POKÉMON_THIS_CARD_IS_A":
                                 target_pokemon.hp += 20
                                 target_pokemon.current_hp += 20
                                 print(f"  Opponent's {target_pokemon.name} HP increased by 20 due to Giant Cape.")
                             action_executed = True
                             reward += 0.05
                         elif target_pokemon is None:
                              print(f"Error: Could not find target {target_type} {target_index_str or ''} for opponent tool.")
                              reward -= 0.5
                              action_executed = False
                         else:
                              print(f"Opponent cannot attach {ghost_card.name}: Target invalid or already has a tool.")
                              reward -= 0.5
                              action_executed = False

                     elif ghost_card is None:
                         print(f"Error: Could not instantiate opponent tool '{card_name}'.")
                         reward -= 0.5
                         action_executed = False
                     else:
                         print(f"Error: Opponent tried to attach non-Tool card '{card_name}'.")
                         reward -= 0.5
                         action_executed = False
                 except Exception as e:
                     print(f"Error parsing opponent attach tool action '{action}': {e}")
                     reward -= 0.5
                     action_executed = False

            elif action.startswith(ACTION_ATTACH_ENERGY_BENCH_PREFIX):
                if is_game_first_turn_overall: # Use the flag checked at the start of step
                     print("Error: Cannot attach energy on the first turn of the game.")
                     reward -= 0.5 # Increased penalty
                     # action_executed remains False
                else:
                    can_attach_energy_this_turn = not self.actions_this_turn.get("energy_attached", False)
                    energy_to_attach = player.energy_stand_available

                    if energy_to_attach and can_attach_energy_this_turn:
                        try:
                            bench_index = int(action.split('_')[-1])
                            if 0 <= bench_index < len(player.bench):
                                 target_pokemon = player.bench[bench_index]
                                 if target_pokemon and not target_pokemon.is_fainted:
                                      # Check energy count BEFORE attaching
                                      current_energy_count = sum(target_pokemon.attached_energy.values())
                                      reward += 0.03
                                      target_pokemon.attach_energy(energy_to_attach)
                                      player.energy_stand_available = None # Consume energy
                                      self.actions_this_turn["energy_attached"] = True
                                      action_executed = True # Attaching energy doesn't end turn
                                      # Reward is now set above based on condition

                                      # --- Check for Nightmare Aura (Darkrai ex) ---
                                      if target_pokemon.name == "Darkrai ex" and energy_to_attach == "Darkness":
                                         print(f"Ability Trigger: {target_pokemon.name}'s Nightmare Aura!")
                                         opponent = self.game_state.get_opponent()
                                         if opponent.active_pokemon and not opponent.active_pokemon.is_fainted:
                                             opponent.active_pokemon.take_damage(20)
                                             print(f"  Nightmare Aura dealt 20 damage to {opponent.active_pokemon.name}.")
                                             # Check for KO from ability damage
                                             if opponent.active_pokemon.is_fainted:
                                                 points_scored = 2 if opponent.active_pokemon.is_ex else 1
                                                 print(f"  {opponent.active_pokemon.name} fainted from Nightmare Aura. {player.name} scores {points_scored} point(s).")
                                                 player.add_point(points_scored)
                                                 reward += 0.75 * points_scored
                                                 winner = self.game_state.check_win_condition()
                                                 if winner:
                                                     print(f"Game Over. Winner: {winner.name}")
                                                     reward += 2.0 # Big reward for winning
                                                     done = True
                                                 elif not opponent.promote_bench_pokemon():
                                                      print(f"Game Over. {opponent.name} has no Pokemon to promote. Winner: {player.name}")
                                                      reward += 2.0 # Reward for winning
                                                      done = True

                                 else:
                                     print(f"Cannot attach energy to bench {bench_index}: Pokemon fainted or invalid.")
                                     reward -= 0.5 # Increased penalty
                                     # action_executed remains False
                            else:
                                 print(f"Invalid bench index in action: {action}")
                                 reward -= 0.5 # Increased penalty
                                 # action_executed remains False
                        except (ValueError, IndexError):
                             print(f"Invalid attach energy bench action format: {action}")
                             reward -= 0.5 # Increased penalty
                             # action_executed remains False
                    else:
                         print(f"Cannot attach energy (Available: {energy_to_attach}, Attached This Turn: {not can_attach_energy_this_turn}).")
                         # Keep penalty for trying illegal attach
                         if energy_to_attach and not can_attach_energy_this_turn:
                             reward -= 0.5 # Increased penalty
                         # action_executed remains False

            # Handle attacking
            elif action.startswith(ACTION_ATTACK_PREFIX): # Corrected indentation
                if is_game_first_turn_overall: # Use the flag checked at the start of step
                    print("Error: Cannot attack on the first turn of the game.")
                    reward -= 0.5 # Increased penalty
                    # action_executed remains False
                else:
                    if player.active_pokemon and not player.active_pokemon.is_fainted:
                        try:
                            attack_index = int(action.split('_')[-1])
                            if 0 <= attack_index < len(player.active_pokemon.attacks):
                                if player.active_pokemon.can_attack(attack_index):
                                    # --- Execute Attack ---
                                    attack = player.active_pokemon.attacks[attack_index]
                                    print(f"{player.name}'s {player.active_pokemon.name} uses {attack.name}!")

                                    # Apply Red's effect if active
                                    damage_modifier = 20 if self.actions_this_turn.get("red_effect_active", False) else 0
                                    print(f"  (Red's effect active: {self.actions_this_turn.get('red_effect_active', False)}, Damage Mod: +{damage_modifier})")

                                    # Calculate damage (including modifier)
                                    damage_dealt = player.active_pokemon.perform_attack(
                                        attack_index,
                                        opponent.active_pokemon,
                                        damage_modifier=damage_modifier
                                    )
                                    reward += damage_dealt * 0.03 # Reward based on damage dealt

                                    # --- Apply Recoil Damage (e.g., Chaotic Impact) ---
                                    if player.active_pokemon.name == "Giratina ex" and attack.name == "Chaotic Impact":
                                        recoil_damage = 20
                                        print(f"  {player.active_pokemon.name} took {recoil_damage} recoil damage from Chaotic Impact.")
                                        player.active_pokemon.take_damage(recoil_damage)
                                        # Check if attacker fainted from recoil
                                        if player.active_pokemon.is_fainted:
                                            points_scored_by_opponent = 2 if player.active_pokemon.is_ex else 1
                                            print(f"  {player.active_pokemon.name} fainted from recoil! {opponent.name} scores {points_scored_by_opponent} point(s).")
                                            opponent.add_point(points_scored_by_opponent)
                                            reward -= 0.75 * points_scored_by_opponent # Negative reward for self-KO
                                            winner = self.game_state.check_win_condition()
                                            if winner:
                                                print(f"Game Over due to recoil KO. Winner: {winner.name}")
                                                reward -= 2.0 # Big penalty for losing via recoil
                                                done = True
                                                # Return early if game ends due to recoil KO
                                                # Need to get state before returning
                                                final_state_for_acting_player = self.get_state_representation(player)
                                                # We need to return from the step function here
                                                # This requires restructuring or a flag, let's use a flag for now
                                                # Or maybe just let the normal turn end logic handle it if done is set?
                                                # Let's set done and let the end-of-step logic handle return.
                                            elif not player.promote_bench_pokemon():
                                                print(f"Game Over. {player.name} fainted from recoil and has no Pokemon to promote. Winner: {opponent.name}")
                                                reward -= 2.0 # Penalty for losing
                                                done = True
                                            # If player promoted successfully, the turn ends normally after opponent KO check

                                    # --- Check for Opponent KO (after potential recoil) ---
                                    # Ensure opponent's active wasn't already handled by recoil KO ending the game
                                    if not done and opponent.active_pokemon and opponent.active_pokemon.is_fainted:
                                        points_scored = 2 if opponent.active_pokemon.is_ex else 1
                                        print(f"  {opponent.active_pokemon.name} fainted! {player.name} scores {points_scored} point(s).")
                                        player.add_point(points_scored)
                                        reward += 0.75 * points_scored
                                        # Check win condition immediately after scoring points
                                        winner = self.game_state.check_win_condition()
                                        if winner:
                                            print(f"Game Over. Winner: {winner.name}")
                                            reward += 2.0 # Big reward for winning
                                            done = True
                                            # No need to promote if game is over
                                        else:
                                            # Opponent needs to promote a new active Pokemon
                                            if not opponent.promote_bench_pokemon():
                                                print(f"Game Over. {opponent.name} has no Pokemon to promote. Winner: {player.name}")
                                                reward += 2.0 # Reward for winning
                                                done = True

                                    action_executed = True # Attack execution ends the turn implicitly later
                                else:
                                    print(f"Cannot use attack {attack_index}: Cannot afford cost.")
                                    reward -= 0.5 # Increased penalty
                            else:
                                print(f"Invalid attack index in action: {action}")
                                reward -= 0.5 # Increased penalty
                        except (ValueError, IndexError):
                            print(f"Invalid attack action format: {action}")
                            reward -= 0.5 # Increased penalty
                    else:
                        print("Cannot attack: No active Pokemon or it's fainted.")
                        reward -= 0.5 # Increased penalty
                        # action_executed remains False

            # --- Handle Target-Specific Item: Potion ---
            elif action.startswith(ACTION_PLAY_ITEM_POTION_TARGET_PREFIX):
                try:
                    parts = action.replace(ACTION_PLAY_ITEM_POTION_TARGET_PREFIX, "").split('_')
                    hand_index = int(parts[0])
                    target_id = "_".join(parts[1:]) # Handles "active" or "bench_X"

                    if 0 <= hand_index < len(player.hand):
                        card_to_play = player.hand[hand_index]
                        if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Item" and card_to_play.effect_tag == "TRAINER_ITEM_POTION_HEAL_20_DAMAGE_FROM_1_OF_YOUR_":
                            print(f"{player.name} plays Item: {card_to_play.name} targeting {target_id}")
                            action_executed = True
                            # Pass target id to effect execution
                            effect_executed_successfully = self._execute_trainer_effect(card_to_play.effect_tag, player, opponent, target_id=target_id)
                            played_card = player.hand.pop(hand_index)
                            player.discard_pile.append(played_card)
                            if effect_executed_successfully: reward += 0.05
                            else: print(f"Effect of {card_to_play.name} failed or had no valid target.")
                        else:
                            print(f"Error: Card at index {hand_index} is not Potion or not an Item.")
                            reward -= 0.5
                    else:
                        print(f"Error: Invalid hand index in action: {action}")
                        reward -= 0.5
                except (ValueError, IndexError):
                    print(f"Error: Invalid Potion action format: {action}")
                    reward -= 0.5

            # Handle playing a Generic Item card (if not target-specific like Potion)
            elif action.startswith(ACTION_PLAY_ITEM_PREFIX):
                try:
                    hand_index = int(action.split('_')[-1])
                    if 0 <= hand_index < len(player.hand):
                        card_to_play = player.hand[hand_index]
                        if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Item":
                            print(f"{player.name} plays Item: {card_to_play.name}")
                            # Mark action as executed *before* effect, as playing the card is valid here
                            action_executed = True

                            # Execute effect
                            effect_executed_successfully = self._execute_trainer_effect(card_to_play.effect_tag, player, opponent)

                            # Remove from hand and discard regardless of effect success
                            played_card = player.hand.pop(hand_index)
                            player.discard_pile.append(played_card)

                            if effect_executed_successfully:
                                reward += 0.05 # Reward for successful effect
                            else:
                                print(f"Effect of {card_to_play.name} failed or had no target.")
                                # No reward for fizzled effect
                        else:
                            print(f"Cannot play card at index {hand_index}: Not an Item card.")
                            reward -= 0.5 # Penalty for trying to play wrong card type
                            # action_executed remains False
                    else:
                        print(f"Error: Invalid hand index in action: {action}")
                        reward -= 0.5 # Add penalty
                        # action_executed remains False
                except (ValueError, IndexError):
                    print(f"Error: Invalid play item action format: {action}")
                    reward -= 0.5 # Add penalty
                    # action_executed remains False

            # Handle playing a Basic Pokemon to the bench
            elif action.startswith(ACTION_PLAY_BASIC_BENCH_PREFIX):
                 if player.can_place_on_bench():
                     try:
                         hand_index = int(action.split('_')[-1])
                         if 0 <= hand_index < len(player.hand):
                             card_to_play = player.hand[hand_index]
                             if isinstance(card_to_play, PokemonCard) and card_to_play.is_basic:
                                 # Move card from hand to bench
                                 played_card = player.hand.pop(hand_index)
                                 player.place_on_bench(played_card)

                                 action_executed = True
                                 reward += 0.02 # Small reward for benching
                             else:
                                 print(f"Cannot play card at index {hand_index}: Not a Basic Pokemon.")
                                 reward -= 0.5 # Increased penalty
                         else:
                             print(f"Invalid hand index in action: {action}")
                             reward -= 0.5 # Increased penalty
                     except (ValueError, IndexError):
                             print(f"Invalid play basic bench action format: {action}")
                             reward -= 0.5 # Increased penalty
                 else:
                     print("Cannot play to bench: Bench is full.")
                     reward -= 0.5 # Increased penalty

            # Handle using an active ability
            elif action == ACTION_USE_ABILITY_ACTIVE:
                source_pokemon = player.active_pokemon
                if source_pokemon and source_pokemon.ability:
                    ability_data = source_pokemon.ability
                    ability_name = ability_data.get("name")
                    ability_type = ability_data.get("type") # Get type from parsed data
                    # ability_type = ability_data.get("type") # REMOVED Duplicate line
                    ability_used_key = f"ability_used_{ability_name}" # Define key
                    if ability_type == "Active" and not self.actions_this_turn.get(ability_used_key, False):
                        print(f"{player.name} uses ability from Active: {ability_name}")
                        # --- Execute Ability Effect ---
                        effect_tag = ability_data.get("effect_tag")
                        # Pass the source pokemon to the effect execution
                        action_executed = True # Mark action as executed for attempting the ability
                        effect_executed, ends_turn = self._execute_ability_effect(effect_tag, player, opponent, source_pokemon)
                        if effect_executed:
                            self.actions_this_turn[ability_used_key] = True # Mark ability as used *if effect succeeded*
                            reward += 0.0
                            if ends_turn:
                                print(f"Ability {ability_name} ends the turn.")
                                action = ACTION_PASS # Treat it like a pass to trigger turn end logic later
                                # Removed immediate return to allow function to complete normally
                        else: # Effect failed
                            print(f"Failed to execute effect for ability {ability_name} (tag: {effect_tag})")
                            reward -= 0.5 # Increased penalty for failed effect
                            # action_executed is already True, so turn proceeds/ends normally
                    else:
                        print(f"Cannot use ability {ability_name} (Not Active type or already used).")
                        reward -= 0.5 # Increased penalty
                else:
                    print("Cannot use ability: Active Pokemon has no ability or cannot use it.")
                    reward -= 0.5 # Increased penalty

            # Handle using a bench ability
            elif action.startswith(ACTION_USE_ABILITY_BENCH_PREFIX):
                try:
                    bench_index = int(action.split('_')[-1])
                    if 0 <= bench_index < len(player.bench):
                        source_pokemon = player.bench[bench_index]
                        if source_pokemon and source_pokemon.ability:
                            ability_data = source_pokemon.ability
                            ability_name = ability_data.get("name")
                            ability_type = ability_data.get("type")
                            ability_used_key = f"ability_used_{ability_name}"
                            if ability_type == "Active" and not self.actions_this_turn.get(ability_used_key, False):
                                print(f"{player.name} uses ability from Bench {bench_index}: {ability_name}")
                                # --- Execute Ability Effect ---
                                effect_tag = ability_data.get("effect_tag")
                                # Pass the source pokemon to the effect execution
                                action_executed = True # Mark action as executed for attempting the ability
                                effect_executed, ends_turn = self._execute_ability_effect(effect_tag, player, opponent, source_pokemon)
                                if effect_executed:
                                    self.actions_this_turn[ability_used_key] = True  # Mark ability as used *if effect succeeded*
                                    # Define energy count for reward calculation
                                    current_energy_count = sum(source_pokemon.attached_energy.values())
                                    reward += 0.04
                                    # Check if the ability ends the turn
                                    if ends_turn:
                                        print(f"Ability {ability_name} ends the turn.")
                                        action = ACTION_PASS  # Treat it like a pass to trigger turn end logic later
                                        # Removed immediate return to allow function to complete normally
                                else: # Effect failed
                                    print(f"Failed to execute effect for ability {ability_name} (tag: {effect_tag})")
                                    reward -= 0.5 # Increased penalty for failed effect
                                    # action_executed is already True, so turn proceeds/ends normally
                            else:
                                print(f"Cannot use ability {ability_name} (Not Active type or already used).")
                                reward -= 0.5 # Increased penalty
                        else:
                            print(f"Cannot use ability: Bench Pokemon at index {bench_index} has no ability or cannot use it.")
                            reward -= 0.5 # Increased penalty
                    else:
                        print(f"Invalid bench index in action: {action}")
                        reward -= 0.5 # Increased penalty
                except (ValueError, IndexError):
                    print(f"Invalid use bench ability action format: {action}")
                    reward -= 0.5 # Increased penalty

            # Handle playing a Supporter card
            elif action.startswith(ACTION_PLAY_SUPPORTER_PREFIX):
                # Removed is_game_first_turn check here based on user clarification
                can_play_supporter = not self.actions_this_turn.get("supporter_played", False)
                if not can_play_supporter:
                    print("Error: Cannot play Supporter: Already played one this turn.")
                    reward -= 0.5 # Add penalty
                    # action_executed remains False
                else:
                    try:
                        hand_index = int(action.split('_')[-1])
                        if 0 <= hand_index < len(player.hand):
                            card_to_play = player.hand[hand_index]
                            if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Supporter":
                                print(f"{player.name} plays Supporter: {card_to_play.name}")
                                # --- MODIFICATION START ---
                                # Mark action as executed *before* effect, as playing the card is valid here
                                action_executed = True
                                self.actions_this_turn["supporter_played"] = True # Mark supporter as played for the turn

                                # Execute effect
                                effect_executed_successfully = self._execute_trainer_effect(card_to_play.effect_tag, player, opponent)

                                # Remove from hand and discard regardless of effect success
                                played_card = player.hand.pop(hand_index)
                                player.discard_pile.append(played_card)

                                if effect_executed_successfully:
                                    reward += 0.05 # Reward for successful effect
                                else:
                                    print(f"Effect of {card_to_play.name} failed or had no target.")
                                    # No reward for fizzled effect, don't penalize as invalid action
                                # --- MODIFICATION END ---
                            else:
                                print(f"Error: Cannot play card at index {hand_index}: Not a Supporter card.")
                                reward -= 0.5 # Add penalty
                                # action_executed remains False
                        else:
                            print(f"Error: Invalid hand index in action: {action}")
                            reward -= 0.5 # Add penalty
                            # action_executed remains False
                    except (ValueError, IndexError):
                        print(f"Error: Invalid play supporter action format: {action}")
                        reward -= 0.5 # Add penalty
                        # No penalty, action_executed remains False
                        pass # Add pass to fix syntax

            # --- Handle Target-Specific Supporter: Cyrus ---
            elif action.startswith(ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX):
                can_play_supporter = not self.actions_this_turn.get("supporter_played", False)
                if not can_play_supporter:
                    print("Error: Cannot play Supporter: Already played one this turn.")
                    reward -= 0.5
                else:
                    try:
                        parts = action.replace(ACTION_PLAY_SUPPORTER_CYRUS_TARGET_PREFIX, "").split('_')
                        hand_index = int(parts[0])
                        target_bench_index = int(parts[1])

                        if 0 <= hand_index < len(player.hand):
                            card_to_play = player.hand[hand_index]
                            if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Supporter" and card_to_play.effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
                                print(f"{player.name} plays Supporter: {card_to_play.name} targeting opponent bench {target_bench_index}")
                                action_executed = True
                                self.actions_this_turn["supporter_played"] = True
                                # Pass target index to effect execution (will be updated later)
                                effect_executed_successfully = self._execute_trainer_effect(card_to_play.effect_tag, player, opponent, target_bench_index=target_bench_index)
                                played_card = player.hand.pop(hand_index)
                                player.discard_pile.append(played_card)
                                if effect_executed_successfully: reward += 0.05
                                else: print(f"Effect of {card_to_play.name} failed or had no valid target.")
                            else:
                                print(f"Error: Card at index {hand_index} is not Cyrus or not a Supporter.")
                                reward -= 0.5
                        else:
                            print(f"Error: Invalid hand index in action: {action}")
                            reward -= 0.5
                    except (ValueError, IndexError):
                        print(f"Error: Invalid Cyrus action format: {action}")
                        reward -= 0.5

            # --- Handle Target-Specific Supporter: Pokemon Center Lady ---
            elif action.startswith(ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX):
                can_play_supporter = not self.actions_this_turn.get("supporter_played", False)
                if not can_play_supporter:
                    print("Error: Cannot play Supporter: Already played one this turn.")
                    reward -= 0.5
                else:
                    try:
                        parts = action.replace(ACTION_PLAY_SUPPORTER_PCL_TARGET_PREFIX, "").split('_')
                        hand_index = int(parts[0])
                        target_id = "_".join(parts[1:]) # Handles "active" or "bench_X"

                        if 0 <= hand_index < len(player.hand):
                            card_to_play = player.hand[hand_index]
                            if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Supporter" and card_to_play.effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
                                print(f"{player.name} plays Supporter: {card_to_play.name} targeting {target_id}")
                                action_executed = True
                                self.actions_this_turn["supporter_played"] = True
                                # Pass target id to effect execution (will be updated later)
                                effect_executed_successfully = self._execute_trainer_effect(card_to_play.effect_tag, player, opponent, target_id=target_id)
                                played_card = player.hand.pop(hand_index)
                                player.discard_pile.append(played_card)
                                if effect_executed_successfully: reward += 0.05
                                else: print(f"Effect of {card_to_play.name} failed or had no valid target.")
                            else:
                                print(f"Error: Card at index {hand_index} is not PCL or not a Supporter.")
                                reward -= 0.5
                        else:
                            print(f"Error: Invalid hand index in action: {action}")
                            reward -= 0.5
                    except (ValueError, IndexError):
                        print(f"Error: Invalid PCL action format: {action}")
                        reward -= 0.5

            # --- Handle Target-Specific Supporter: Dawn ---
            elif action.startswith(ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX):
                can_play_supporter = not self.actions_this_turn.get("supporter_played", False)
                if not can_play_supporter:
                    print("Error: Cannot play Supporter: Already played one this turn.")
                    reward -= 0.5
                else:
                    try:
                        parts = action.replace(ACTION_PLAY_SUPPORTER_DAWN_SOURCE_TARGET_PREFIX, "").split('_')
                        hand_index = int(parts[0])
                        source_bench_index = int(parts[1])

                        if 0 <= hand_index < len(player.hand):
                            card_to_play = player.hand[hand_index]
                            if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Supporter" and card_to_play.effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
                                print(f"{player.name} plays Supporter: {card_to_play.name} sourcing from bench {source_bench_index}")
                                action_executed = True
                                self.actions_this_turn["supporter_played"] = True
                                # Pass source index to effect execution (will be updated later)
                                effect_executed_successfully = self._execute_trainer_effect(card_to_play.effect_tag, player, opponent, source_bench_index=source_bench_index)
                                played_card = player.hand.pop(hand_index)
                                player.discard_pile.append(played_card)
                                if effect_executed_successfully: reward += 0.05
                                else: print(f"Effect of {card_to_play.name} failed or had no valid target/source.")
                            else:
                                print(f"Error: Card at index {hand_index} is not Dawn or not a Supporter.")
                                reward -= 0.5
                        else:
                            print(f"Error: Invalid hand index in action: {action}")
                            reward -= 0.5
                    except (ValueError, IndexError):
                        print(f"Error: Invalid Dawn action format: {action}")
                        reward -= 0.5

            # Handle playing a Generic Item card (if not target-specific)
            elif action.startswith(ACTION_PLAY_ITEM_PREFIX): # Removed the 'and not potion' check as it's handled above
                try:
                    hand_index = int(action.split('_')[-1])
                    if 0 <= hand_index < len(player.hand):
                        card_to_play = player.hand[hand_index]
                        if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Item":
                            print(f"{player.name} plays Item: {card_to_play.name}")
                            # --- MODIFICATION START ---
                            # Mark action as executed *before* effect, as playing the card is valid here
                            action_executed = True

                            # Execute effect
                            effect_executed_successfully = self._execute_trainer_effect(card_to_play.effect_tag, player, opponent)

                            # Remove from hand and discard regardless of effect success
                            played_card = player.hand.pop(hand_index)
                            player.discard_pile.append(played_card)

                            if effect_executed_successfully:
                                reward += 0.05 # Reward for successful effect
                            else:
                                print(f"Effect of {card_to_play.name} failed or had no target.")
                                # No reward for fizzled effect
                            # --- MODIFICATION END ---
                        else:
                            print(f"Cannot play card at index {hand_index}: Not an Item card.")
                            reward -= 0.5 # Penalty for trying to play wrong card type
                            # action_executed remains False
                    else:
                        print(f"Error: Invalid hand index in action: {action}")
                        reward -= 0.5 # Add penalty
                        # action_executed remains False
                except (ValueError, IndexError):
                    print(f"Error: Invalid play item action format: {action}")
                    reward -= 0.5 # Add penalty
                    # action_executed remains False

            # Handle attaching a Tool card
            elif action.startswith(ACTION_ATTACH_TOOL_ACTIVE):
                try:
                    hand_index = int(action.split('_')[-1])
                    if 0 <= hand_index < len(player.hand):
                        card_to_play = player.hand[hand_index]
                        target_pokemon = player.active_pokemon
                        if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Tool":
                            if target_pokemon and not target_pokemon.is_fainted and target_pokemon.attached_tool is None:
                                # Attach the tool
                                tool_card = player.hand.pop(hand_index)
                                target_pokemon.attached_tool = tool_card
                                print(f"{player.name} attached {tool_card.name} to active {target_pokemon.name}.")
                                # Apply immediate effects (e.g., HP boost)
                                if tool_card.effect_tag == "TRAINER_TOOL_GIANT_CAPE_THE_POK_MON_THIS_CARD_IS_ATTACHED_TO_G":
                                    target_pokemon.hp += 20
                                    target_pokemon.current_hp += 20 # Also increase current HP
                                    print(f"  {target_pokemon.name} HP increased by 20 due to Giant Cape.")
                                action_executed = True
                                reward += 0.05 # Increased reward for attaching tool
                            else:
                                print(f"Cannot attach {card_to_play.name}: Target invalid or already has a tool.")
                                reward -= 0.5 # Increased penalty
                        else:
                            print(f"Cannot attach card at index {hand_index}: Not a Tool card.")
                            reward -= 0.5 # Increased penalty
                    else:
                        print(f"Invalid hand index in action: {action}")
                        reward -= 0.5 # Increased penalty
                except (ValueError, IndexError):
                    print(f"Invalid attach tool active action format: {action}")
                    reward -= 0.5 # Increased penalty
            elif action.startswith(ACTION_ATTACH_TOOL_BENCH_PREFIX):
                try:
                    parts = action.split('_')
                    bench_index = int(parts[-2])
                    hand_index = int(parts[-1])

                    if 0 <= hand_index < len(player.hand) and 0 <= bench_index < len(player.bench):
                        card_to_play = player.hand[hand_index]
                        target_pokemon = player.bench[bench_index]
                        if isinstance(card_to_play, TrainerCard) and card_to_play.trainer_type == "Tool":
                            if target_pokemon and not target_pokemon.is_fainted and target_pokemon.attached_tool is None:
                                # Attach the tool
                                tool_card = player.hand.pop(hand_index)
                                target_pokemon.attached_tool = tool_card
                                print(f"{player.name} attached {tool_card.name} to benched {target_pokemon.name} (Index {bench_index}).")
                                # Apply immediate effects (e.g., HP boost)
                                if tool_card.effect_tag == "TRAINER_TOOL_GIANT_CAPE_THE_POK_MON_THIS_CARD_IS_ATTACHED_TO_G":
                                    target_pokemon.hp += 20
                                    target_pokemon.current_hp += 20 # Also increase current HP
                                    print(f"  {target_pokemon.name} HP increased by 20 due to Giant Cape.")
                                action_executed = True
                                reward += 0.05 # Increased reward for attaching tool
                            else:
                                print(f"Cannot attach {card_to_play.name}: Target invalid or already has a tool.")
                                reward -= 0.5 # Increased penalty
                        else:
                            print(f"Cannot attach card at index {hand_index}: Not a Tool card.")
                            reward -= 0.5 # Increased penalty
                    else:
                        print(f"Invalid bench/hand index in action: {action}")
                        reward -= 0.5 # Increased penalty
                except (ValueError, IndexError):
                    print(f"Invalid attach tool bench action format: {action}")
                    reward -= 0.5 # Increased penalty

            # Handle retreating
            elif action.startswith(ACTION_RETREAT_TO_BENCH_PREFIX):
                can_retreat_this_turn = not self.actions_this_turn.get("retreat_used", False)
                if not can_retreat_this_turn:
                    print("Cannot retreat: Already retreated this turn.")
                    reward -= 0.5 # Increased penalty
                else:
                    try:
                        bench_index = int(action.split('_')[-1])
                        if 0 <= bench_index < len(player.bench):
                            target_bench_pokemon = player.bench[bench_index]
                            current_active_pokemon = player.active_pokemon

                            # Check if both active and target bench pokemon are valid for swap
                            if current_active_pokemon and not current_active_pokemon.is_fainted and \
                            target_bench_pokemon and not target_bench_pokemon.is_fainted:

                                # Verify retreat cost can be paid again (safety check)
                                retreat_cost_modifier = -2 if self.actions_this_turn.get("leaf_effect_active", False) else 0
                                if current_active_pokemon.can_retreat(current_turn_cost_modifier=retreat_cost_modifier):
                                    effective_cost = max(0, current_active_pokemon.retreat_cost + retreat_cost_modifier)
                                    print(f"{player.name} retreats {current_active_pokemon.name} (Cost: {effective_cost}) to bench, promotes {target_bench_pokemon.name} (Index {bench_index}).")

                                    # Discard energy for retreat cost
                                    current_active_pokemon.discard_energy(effective_cost)

                                    # Perform the swap
                                    player.active_pokemon = target_bench_pokemon
                                    player.bench[bench_index] = current_active_pokemon # Place old active onto bench slot

                                    # Update flags
                                    player.active_pokemon.is_active = True
                                    current_active_pokemon.is_active = False

                                    # Mark retreat as used for the turn
                                    self.actions_this_turn["retreat_used"] = True
                                    action_executed = True
                                    reward += 0.01 # Small reward for successful retreat

                                    # Retreat does not end the turn.
                                else:
                                    print(f"Cannot retreat: Cannot afford cost (Effective: {max(0, current_active_pokemon.retreat_cost + retreat_cost_modifier)}, Attached: {sum(current_active_pokemon.attached_energy.values())}).")
                                    reward -= 0.5 # Increased penalty
                            else:
                                print(f"Cannot retreat: Invalid active or target bench Pokemon (Index {bench_index}).")
                                reward -= 0.5 # Increased penalty
                        else:
                            print(f"Invalid bench index in retreat action: {action}")
                            reward -= 0.5 # Increased penalty
                    except (ValueError, IndexError):
                        print(f"Invalid retreat action format: {action}")
                        reward -= 0.5 # Increased penalty

        else: # This is the final else for unknown actions
            print(f"Unknown action attempted: {action}") # Changed message slightly
            reward = -0.5
            action_executed = False # Explicitly false
            action_was_known = False # Mark as unknown

        # Define if the action is turn-ending (needed for logic below)
        is_turn_ending_action = action == ACTION_PASS or action.startswith(ACTION_ATTACK_PREFIX)
        # Note: Abilities ending the turn often set action=ACTION_PASS internally, handled by is_turn_ending_action

        # --- Post-Action Processing & Turn End Logic ---
        # This section now only runs for:
        # 1. Invalid actions (action_executed is False)
        # 2. Turn-ending actions (PASS, ATTACK) during a normal turn
        # 3. Actions during the setup phase (which handle their own returns/turn switches)

        # Check for game end immediately after any action that might cause it (KO, deck out)
        # Note: Win condition checks are already inside attack/ability logic where KOs happen.
        # We might need a deck-out check here if drawing cards can happen via effects.
        if done: # If a win condition was met during the action execution
            final_state_for_acting_player = self.get_state_representation(player)
            return final_state_for_acting_player, reward, done

        # Determine if the game turn should *actually* switch
        switch_game_turn = False
        if not is_setup_phase: # Only consider turn switching during normal play
            if action_executed and is_turn_ending_action:
                # Switch turn only if a successful turn-ending action was executed
                print("--- End of Turn ---")
                # TODO: Implement end-of-turn checks (poison, burn, etc.) if needed
                switch_game_turn = True
            elif not action_was_known:
                 # Switch turn if the action was completely unknown
                 print(f"Unknown action '{action}' attempted. Ending turn.")
                 switch_game_turn = True
            # else: # Case: action_executed is False AND action_was_known is True
                 # Known action failed internal checks (e.g., playing 2nd supporter, cannot afford cost).
                 # The specific error was printed within the action block.
                 # DO NOT switch turn. The player gets another chance.
                 # The function will proceed to the final return block below.
                 pass # Explicitly do nothing here

        # If the game turn is switching (due to successful PASS/ATTACK or UNKNOWN action), handle the switch
        if switch_game_turn:
            self.game_state.switch_turn() # Switches player index and increments turn number if needed

            # Check turn limit after switching
            if self.game_state.turn_number > self.turn_limit:
                print("Game Over! Turn limit reached.")
                done = True
                reward = -1.0 # Penalty for reaching turn limit
                # If done by turn limit, the state for the *next* player is less relevant,
                # but we still need to return something. Get state for the player who just acted.
                final_state_for_acting_player = self.get_state_representation(player)
                return final_state_for_acting_player, reward, done
            else:
                # Start the next player's turn (draw card, update energy stand)
                self._start_turn()

        # --- Get Next State ---
        # This part is reached if:
        # a) An invalid action occurred (turn switches)
        # b) A turn-ending action occurred (turn switches)
        # c) A setup action occurred that didn't immediately return (e.g., confirm ready but opponent not ready) - turn might switch here too.
        # The observation should always be from the perspective of the player whose turn it is *now*.
        next_player = self.game_state.get_current_player()
        next_state = self.get_state_representation(next_player)

        # Return the observation for the current player, reward, and done status.
        return next_state, reward, done # This is the final return if the early return wasn't triggered


    def _commit_setup_choices(self):
        """Simultaneously moves pending setup cards from hand to active/bench for both players."""
        print("\n--- Committing Setup Choices ---")
        players_to_commit = [self.player1, self.player2]
        all_cards_to_remove_from_hand = {p.name: [] for p in players_to_commit} # Store instances

        # 1. Assign Active Pokemon
        for player in players_to_commit:
            if player.pending_setup_active:
                player.active_pokemon = player.pending_setup_active
                player.active_pokemon.is_active = True
                all_cards_to_remove_from_hand[player.name].append(player.pending_setup_active) # Add instance
                print(f"{player.name} reveals Active: {player.active_pokemon.name}")
            else:
                print(f"CRITICAL ERROR: Player {player.name} confirmed ready without a pending active!")
                # Handle error - maybe force mulligan or raise exception?

        # 2. Assign Bench Pokemon
        for player in players_to_commit:
            player.bench = list(player.pending_setup_bench) # Assign the list of pending cards
            for bench_pokemon in player.bench:
                bench_pokemon.is_active = False
                all_cards_to_remove_from_hand[player.name].append(bench_pokemon) # Add instances
            bench_names = ", ".join(p.name for p in player.bench) if player.bench else "None"
            print(f"{player.name} reveals Bench ({len(player.bench)}): [{bench_names}]")

        # 3. Remove all chosen cards from hands simultaneously using instance comparison
        for player in players_to_commit:
            cards_to_remove_instances = all_cards_to_remove_from_hand[player.name]
            original_hand = list(player.hand) # Copy original hand
            new_hand = []
            removed_count = 0
            expected_remove_count = len(cards_to_remove_instances)

            # Iterate through original hand and keep cards NOT in the removal list
            for card_in_hand in original_hand:
                is_card_to_remove = False
                for card_to_remove in cards_to_remove_instances:
                    if card_in_hand is card_to_remove: # Direct instance comparison
                        is_card_to_remove = True
                        removed_count += 1
                        break
                if not is_card_to_remove:
                    new_hand.append(card_in_hand)

            # Check if the correct number of cards were identified for removal
            if removed_count != expected_remove_count:
                 print(f"Warning: Mismatch removing setup cards from {player.name}'s hand. Expected {expected_remove_count}, identified {removed_count} for removal.")
                 # Fallback: Remove by name if instance matching failed (less reliable with duplicates)
                 # This part needs careful consideration if duplicates are common and instance tracking fails
                 # For now, proceed with the potentially incorrect new_hand based on instance match

            player.hand = new_hand # Assign the filtered hand
            print(f"{player.name} Hand size after setup commit: {len(player.hand)}")


        # 4. Clear pending attributes
        for player in players_to_commit:
            player.pending_setup_active = None
            player.pending_setup_bench = []
            # Keep setup_ready = True

        # 5. Ensure the current player index is reset to the original starting player
        # This is crucial because setup actions might have flipped the index temporarily.
        self.game_state.current_player_index = self.game_state.starting_player_index
        print(f"--- Setup Commit Complete (Current player reset to starter: {self.game_state.get_current_player().name}) ---")


    def _execute_ability_effect(self, effect_tag: Optional[str], player: Player, opponent: Player, source_pokemon: PokemonCard) -> Tuple[bool, bool]:
        """
        Executes the game logic for a given ability effect tag, originating from source_pokemon.
        Returns a tuple: (effect_executed: bool, ends_turn: bool)
        """
        if not effect_tag:
            print("Warning: Ability has no effect tag.")
            return False, False # Failed execution, doesn't end turn

        print(f"Executing ability effect for tag: {effect_tag}")
        ends_turn = False # Default: ability does not end turn

        # --- Darkrai ex - Nightmare Aura (Passive - should not be actively used) ---
        if effect_tag == "ABILITY_NIGHTMARE_AURA_WHENEVER_YOU_ATTACH_A_D_ENERGY_FROM_YOUR":
            print(f"Warning: Nightmare Aura is a passive ability and cannot be actively used.")
            return False, ends_turn # Failed execution

        # --- Giratina ex - Broken-Space Bellow ---
        elif effect_tag == "ABILITY_BROKEN_SPACE_BELLOW_ONCE_DURING_YOUR_TURN_":
            # Ability targets the Pokemon using it (source_pokemon)
            if source_pokemon.name == "Giratina ex": # Ensure it's Giratina using it
                print(f"{player.name} uses {source_pokemon.name}'s Broken-Space Bellow, attaching Psychic energy to itself.")
                source_pokemon.attach_energy("Psychic") # Attach the generated energy to Giratina
                ends_turn = True # This ability ends the turn
                return True, ends_turn
            else:
                # Should not happen if called correctly, but safety check
                print(f"Error: Broken-Space Bellow effect triggered by non-Giratina ({source_pokemon.name}).")
                return False, ends_turn

        # --- Example: Draw Card Ability ---
        # elif effect_tag == "DRAW_CARD_1": # Example tag
        #     player.draw_cards(1)
        #     return True, ends_turn # Executed, doesn't end turn

        # --- Fallback for unhandled tags ---
        else:
            print(f"Warning: Unhandled ability effect tag '{effect_tag}'")
            return False, ends_turn # Failed execution

    def _execute_trainer_effect(self, effect_tag: Optional[str], player: Player, opponent: Player,
                                target_bench_index: Optional[int] = None,
                                target_id: Optional[str] = None,
                                source_bench_index: Optional[int] = None) -> bool:
        """
        Executes the game logic for a given trainer card effect tag.
        Accepts optional target/source indices for specific effects.
        Returns True if effect executed successfully, False otherwise.
        """
        # Note: Trainer effects generally don't end the turn unless specified
        if not effect_tag:
            print("Warning: Trainer card has no effect tag.")
            return False # Cannot execute an effect without a tag

        print(f"Executing trainer effect for tag: {effect_tag}")

        # --- Supporter Effects ---
        if effect_tag == "TRAINER_SUPPORTER_PROFESSOR_S_RESEARCH_DRAW_2_CARD":
            player.draw_cards(2)
            return True
        elif effect_tag == "TRAINER_SUPPORTER_RED_DURING_THIS_TURN_ATTACKS_USE":
            print(f"{player.name}'s Pokemon attacks do +20 damage this turn.")
            self.actions_this_turn["red_effect_active"] = True # Set the flag for this turn
            return True # Effect is successfully activated
        elif effect_tag == "TRAINER_SUPPORTER_SABRINA_SWITCH_OUT_YOUR_OPPONENT":
            if opponent.active_pokemon and opponent.bench:
                # In TCG Pocket, opponent chooses the new active
                # For simulation, we might need a strategy or random choice
                # Let's assume random choice for now
                print(f"{player.name} uses Sabrina. {opponent.name} must switch active.")
                benched_indices = [i for i, p in enumerate(opponent.bench) if p and not p.is_fainted]
                if not benched_indices:
                    print(f"{opponent.name} has no valid Pokemon to switch in.")
                    return False # Cannot execute if no valid bench Pokemon

                chosen_bench_index = random.choice(benched_indices)
                new_active = opponent.bench.pop(chosen_bench_index)
                old_active = opponent.active_pokemon
                opponent.active_pokemon = new_active
                opponent.active_pokemon.is_active = True
                # Place old active onto bench if possible
                if opponent.can_place_on_bench():
                    opponent.place_on_bench(old_active)
                    old_active.is_active = False
                else:
                    # If bench is full, the old active is discarded (Pocket rules might differ?)
                    print(f"Warning: {opponent.name}'s bench is full. Discarding {old_active.name}.")
                    opponent.discard_pile.append(old_active)

                print(f"{opponent.name} switched {opponent.active_pokemon.name} to active.")
                return True
            else:
                print(f"Cannot use Sabrina: Opponent has no active or no bench.")
                return False
        elif effect_tag == "TRAINER_SUPPORTER_CYRUS_SWITCH_IN_1_OF_YOUR_OPPONE":
             # Similar to Sabrina, but targets damaged Pokemon and player chooses
             if opponent.active_pokemon:
                 damaged_benched_indices = [i for i, p in enumerate(opponent.bench) if p and not p.is_fainted and p.current_hp < p.hp]
                 if not damaged_benched_indices:
                     print(f"Cannot use Cyrus: Opponent has no damaged benched Pokemon.")
                     return False

                 # --- Use provided target_bench_index ---
                 if target_bench_index is None or target_bench_index not in damaged_benched_indices:
                     print(f"Error: Invalid or missing target_bench_index ({target_bench_index}) for Cyrus. Valid: {damaged_benched_indices}")
                     # Fallback: random choice (should not happen if called correctly)
                     # chosen_bench_index = random.choice(damaged_benched_indices)
                     return False # Fail if target is invalid

                 chosen_bench_index = target_bench_index
                 print(f"{player.name} uses Cyrus, bringing up opponent's bench Pokemon at index {chosen_bench_index}.")

                 # Ensure the chosen index is still valid before popping (safety check)
                 if chosen_bench_index >= len(opponent.bench) or not opponent.bench[chosen_bench_index]:
                     print(f"Error: Target bench index {chosen_bench_index} became invalid before swap.")
                     return False

                 new_active = opponent.bench.pop(chosen_bench_index)
                 old_active = opponent.active_pokemon
                 opponent.active_pokemon = new_active
                 opponent.active_pokemon.is_active = True
                 # Place old active onto bench if possible
                 if opponent.can_place_on_bench():
                     opponent.place_on_bench(old_active)
                     old_active.is_active = False
                 else:
                     print(f"Warning: {opponent.name}'s bench is full. Discarding {old_active.name}.")
                     opponent.discard_pile.append(old_active)
                 print(f"{opponent.name} now has {opponent.active_pokemon.name} as active.")
                 return True
             else:
                 print("Cannot use Cyrus: Opponent has no active Pokemon.")
                 return False
        elif effect_tag == "TRAINER_SUPPORTER_MARS_YOUR_OPPONENT_SHUFFLES_THEI":
             print(f"{player.name} uses Mars.")
             # --- CORRECTED: Target opponent, not player ---
             opponent_hand_size = len(opponent.hand)
             print(f"{opponent.name} shuffles {opponent_hand_size} cards into deck.")
             opponent.deck.extend(opponent.hand) # Add opponent's hand to their deck
             opponent.hand = [] # Clear opponent's hand
             random.shuffle(opponent.deck) # Shuffle opponent's deck
             # Draw based on opponent's remaining points
             points_needed = max(0, POINTS_TO_WIN - opponent.points)
             print(f"{opponent.name} needs {points_needed} points, drawing that many cards.")
             opponent.draw_cards(points_needed) # Opponent draws
             # --- END CORRECTION ---
             return True
        elif effect_tag == "TRAINER_SUPPORTER_POKÉMON_CENTER_LADY_HEAL_30_DAMA":
             # Identify all potential targets (active + benched, not fainted)
             potential_targets = []
             if player.active_pokemon and not player.active_pokemon.is_fainted:
                 potential_targets.append(player.active_pokemon)
             for pokemon in player.bench:
                 if pokemon and not pokemon.is_fainted:
                     potential_targets.append(pokemon)

             if not potential_targets:
                 print("Cannot use Pokemon Center Lady: No valid Pokemon to heal.")
                 return False # Effect fails if no targets

             # --- Use provided target_id ---
             target_pokemon: Optional[PokemonCard] = None
             target_location = "unknown"
             if target_id == "active":
                 if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.current_hp < player.active_pokemon.hp:
                     target_pokemon = player.active_pokemon
                     target_location = "active"
             elif target_id and target_id.startswith("bench_"):
                 try:
                     bench_idx = int(target_id.split('_')[-1])
                     if 0 <= bench_idx < len(player.bench) and player.bench[bench_idx] and not player.bench[bench_idx].is_fainted and player.bench[bench_idx].current_hp < player.bench[bench_idx].hp:
                         target_pokemon = player.bench[bench_idx]
                         target_location = f"benched (index {bench_idx})"
                 except (ValueError, IndexError):
                     pass # Invalid format

             if target_pokemon is None:
                 print(f"Error: Invalid or missing target_id ('{target_id}') for Pokemon Center Lady.")
                 # Fallback: random choice (should not happen if called correctly)
                 # target = random.choice(potential_targets)
                 # target_location = "active" if target.is_active else "benched"
                 return False # Fail if target is invalid

             heal_amount = 30
             healed = min(heal_amount, target_pokemon.hp - target_pokemon.current_hp)
             target_pokemon.current_hp += healed
             # TODO: Implement special condition removal if needed (currently not simulated)
             print(f"{player.name} uses Pokemon Center Lady, healing {target_location} {target_pokemon.name} by {healed} HP.")
             return True
        elif effect_tag == "TRAINER_SUPPORTER_LEAF_DURING_THIS_TURN_THE_RETREA":
             print(f"{player.name} uses Leaf. Active Pokemon's retreat cost is reduced by 2 this turn.")
             self.actions_this_turn["leaf_effect_active"] = True # Set the flag for this turn
             return True # Effect is successfully activated
        elif effect_tag == "TRAINER_SUPPORTER_DAWN_MOVE_AN_ENERGY_FROM_1_OF_YO":
             # Needs source and target selection (AI decision)
             # Find benched pokemon with energy
             source_options = [(i, p) for i, p in enumerate(player.bench) if p and not p.is_fainted and p.attached_energy]
             target = player.active_pokemon
             if not source_options or not target or target.is_fainted:
                 print("Cannot use Dawn: No valid source or target Pokemon.")
                 return False

             # --- Use provided source_bench_index ---
             source_pokemon: Optional[PokemonCard] = None
             if source_bench_index is not None and 0 <= source_bench_index < len(player.bench):
                 potential_source = player.bench[source_bench_index]
                 # Verify the chosen source is valid (has energy, not fainted)
                 if potential_source and not potential_source.is_fainted and potential_source.attached_energy:
                     source_pokemon = potential_source
                 else:
                     print(f"Error: Provided source bench index {source_bench_index} is invalid for Dawn (No Pokemon, fainted, or no energy).")
                     return False # Fail if source is invalid
             else:
                 print(f"Error: Invalid or missing source_bench_index ({source_bench_index}) for Dawn.")
                 # Fallback: random choice (should not happen if called correctly)
                 # source_index, source_pokemon = random.choice(source_options)
                 return False # Fail if source is invalid

             # AI still needs to choose which energy type to move if multiple exist - random for now
             if not source_pokemon.attached_energy: # Should be caught above, but safety check
                 print(f"Error: Source Pokemon {source_pokemon.name} has no energy to move.")
                 return False
             energy_type_to_move = random.choice(list(source_pokemon.attached_energy.keys()))

             print(f"{player.name} uses Dawn, moving 1 {energy_type_to_move} from bench {source_bench_index} ({source_pokemon.name}) to active ({target.name}).")
             source_pokemon.attached_energy[energy_type_to_move] -= 1
             if source_pokemon.attached_energy[energy_type_to_move] == 0:
                 del source_pokemon.attached_energy[energy_type_to_move]
             target.attach_energy(energy_type_to_move)
             return True
        elif effect_tag == "TRAINER_SUPPORTER_TEAM_ROCKET_GRUNT_FLIP_A_COIN_UN":
             print(f"{player.name} uses Team Rocket Grunt.")
             heads_count = 0
             while True:
                 if random.choice([True, False]): # True = Heads
                     heads_count += 1
                     print("Coin flip: Heads")
                 else:
                     print("Coin flip: Tails")
                     break
             print(f"Discarding {heads_count} energy from {opponent.name}'s active Pokemon.")
             if opponent.active_pokemon and not opponent.active_pokemon.is_fainted:
                 for _ in range(heads_count):
                     if not opponent.active_pokemon.attached_energy:
                         print(f"{opponent.active_pokemon.name} has no more energy to discard.")
                         break
                     # Discard random energy
                     energy_type_to_discard = random.choice(list(opponent.active_pokemon.attached_energy.keys()))
                     opponent.active_pokemon.attached_energy[energy_type_to_discard] -= 1
                     print(f"Discarded 1 {energy_type_to_discard} energy.")
                     if opponent.active_pokemon.attached_energy[energy_type_to_discard] == 0:
                         del opponent.active_pokemon.attached_energy[energy_type_to_discard]
             else:
                 print(f"{opponent.name} has no active Pokemon to discard energy from.")
             return True
        elif effect_tag == "TRAINER_SUPPORTER_IONO_EACH_PLAYER_SHUFFLES_THE_CA":
             print(f"{player.name} uses Iono.")
             # Player shuffles and draws
             player_hand_size = len(player.hand)
             print(f"{player.name} shuffles {player_hand_size} cards into deck.")
             player.deck.extend(player.hand)
             player.hand = []
             random.shuffle(player.deck)
             player.draw_cards(player_hand_size)
             # Opponent shuffles and draws
             opponent_hand_size = len(opponent.hand)
             print(f"{opponent.name} shuffles {opponent_hand_size} cards into deck.")
             opponent.deck.extend(opponent.hand)
             opponent.hand = []
             random.shuffle(opponent.deck)
             opponent.draw_cards(opponent_hand_size)
             return True

        # --- Item Effects ---
        elif effect_tag == "TRAINER_ITEM_POKÉ_BALL_PUT_1_RANDOM_BASIC_POKEMON_": # Corrected tag
            print(f"{player.name} uses Poke Ball.")
            basic_pokemon_in_deck = [card for card in player.deck if isinstance(card, PokemonCard) and card.is_basic]
            if not basic_pokemon_in_deck:
                print("No Basic Pokemon found in deck.")
                # Shuffle deck anyway? TCG rules vary. Let's assume shuffle.
                random.shuffle(player.deck)
                return True # Effect considered resolved even if no Pokemon found
            chosen_pokemon = random.choice(basic_pokemon_in_deck)
            player.deck.remove(chosen_pokemon) # Remove from deck
            random.shuffle(player.deck) # Shuffle deck
            player.hand.append(chosen_pokemon) # Add to hand
            print(f"Found {chosen_pokemon.name} and added it to hand.")
            return True
        elif effect_tag == "TRAINER_ITEM_POTION_HEAL_20_DAMAGE_FROM_1_OF_YOUR_": # Corrected tag
             # Identify potential targets (damaged, not fainted)
             potential_targets = []
             if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.current_hp < player.active_pokemon.hp:
                 potential_targets.append(("active", player.active_pokemon))
             for i, p in enumerate(player.bench):
                 if p and not p.is_fainted and p.current_hp < p.hp:
                     potential_targets.append((f"bench_{i}", p))

             if not potential_targets:
                 print("Cannot use Potion: No damaged Pokemon.")
                 return False

             # --- Use provided target_id ---
             target_pokemon: Optional[PokemonCard] = None
             target_location = "unknown"
             if target_id == "active":
                 if player.active_pokemon and not player.active_pokemon.is_fainted and player.active_pokemon.current_hp < player.active_pokemon.hp:
                     target_pokemon = player.active_pokemon
                     target_location = "active"
             elif target_id and target_id.startswith("bench_"):
                 try:
                     bench_idx = int(target_id.split('_')[-1])
                     if 0 <= bench_idx < len(player.bench) and player.bench[bench_idx] and not player.bench[bench_idx].is_fainted and player.bench[bench_idx].current_hp < player.bench[bench_idx].hp:
                         target_pokemon = player.bench[bench_idx]
                         target_location = f"benched (index {bench_idx})"
                 except (ValueError, IndexError):
                     pass # Invalid format

             if target_pokemon is None:
                 print(f"Error: Invalid or missing target_id ('{target_id}') for Potion.")
                 # Fallback: random choice (should not happen if called correctly)
                 # chosen_id, target_pokemon = random.choice(potential_targets)
                 # target_location = chosen_id
                 return False # Fail if target is invalid

             heal_amount = 20
             healed = min(heal_amount, target_pokemon.hp - target_pokemon.current_hp)
             target_pokemon.current_hp += healed
             print(f"{player.name} uses Potion, healing {target_location} {target_pokemon.name} by {healed} HP.")
             return True

         # --- Tool Effects ---
        elif effect_tag == "TRAINER_TOOL_GIANT_CAPE_THE_POKÉMON_THIS_CARD_IS_A":
            print(f"Info: Giant Cape effect applied during attachment.")
            return True # Effect is handled, return True so card is discarded
        elif effect_tag == "TRAINER_TOOL_ROCKY_HELMET_IF_THE_POKÉMON_THIS_CARD":
            print(f"Info: Rocky Helmet effect handled during damage calculation.")
            return True # Effect is handled, return True so card is discarded

        # --- Fallback ---
        else:
            print(f"Warning: Unhandled trainer effect tag '{effect_tag}'")
            return False


    def _instantiate_card_from_name(self, card_name: str) -> Optional[Card]:
        """Instantiates a single Card object based on its name using loaded card_data."""
        card_info = self.card_data.get(card_name)
        if not card_info:
            print(f"Error: Card name '{card_name}' not found in loaded card data during instantiation.")
            return None

        # Use deepcopy to ensure the instance is unique, though it's temporary
        card_info_copy = copy.deepcopy(card_info)
        card_type = card_info_copy.get("card_type")

        if card_type == "Pokemon":
            # We'll need attack instantiation logic if opponent could play evolved Pokemon,
            # but for ACTION_OPP_PLAY_BASIC_, we only need basic info.
                attacks = []
                if card_info_copy.get("attacks"):
                    for attack_data in card_info_copy["attacks"]:
                        attacks.append(Attack(
                        name=attack_data.get("name", "Unknown Attack"),
                        cost=attack_data.get("cost", {}),
                        damage=attack_data.get("damage", 0),
                        effect=attack_data.get("effect_tag")
                    ))
                # Ensure it's basic for the OPP_PLAY_BASIC action
                if not card_info_copy.get("is_basic", False):
                    print(f"Error: Tried to instantiate non-basic Pokemon '{card_name}' for basic play action.")
                    return None
                return PokemonCard(
                    name=card_info_copy.get("name", "Unknown Pokemon"),
                    hp=card_info_copy.get("hp", 0),
                    attacks=attacks, # Include attacks just in case
                    pokemon_type=card_info_copy.get("pokemon_type", "Colorless"),
                    weakness_type=card_info_copy.get("weakness_type"),
                    retreat_cost=card_info_copy.get("retreat_cost", 0),
                    is_ex=card_info_copy.get("is_ex", False),
                    is_basic=True, # Enforce basic
                    ability=card_info_copy.get("ability")
                )
        elif card_type in ["Supporter", "Item", "Tool", "Stadium"]:
            return TrainerCard(
                name=card_info_copy.get("name", "Unknown Trainer"),
                trainer_type=card_type,
                effect_tag=card_info_copy.get("effect_tag")
            )
        elif card_type == "Energy":
            # Handle energy if needed, though opponent doesn't play these from hand usually
            print(f"Warning: Energy card '{card_name}' instantiation not fully handled.")
            # Placeholder - might need an EnergyCard class if they can be played like this
            return Card(name=card_name) # Basic card instance
        else:
            print(f"Warning: Card '{card_name}' has unknown or missing card_type '{card_type}'.")
            return Card(name=card_name) # Basic card instance

    def synchronize_player_state(self, player: Player, actual_hand_card_names: List[str], actual_deck_size: int):
        """
        Forces the simulation's state for a player's hand and deck to match the
        observed state from the real game, typically after an effect resolves.

        Args:
            player: The Player object (either self.player1 or self.player2) to synchronize.
            actual_hand_card_names: A list of exact card names representing the player's
                                     hand in the real game *after* the effect.
            actual_deck_size: The number of cards remaining in the player's deck in the
                              real game *after* the effect.
        """
        print(f"\n--- Synchronizing {player.name}'s State ---")
        print(f"Target Hand ({len(actual_hand_card_names)} cards): {', '.join(actual_hand_card_names)}")
        print(f"Target Deck Size: {actual_deck_size}")

        # 1. Combine all known card instances for the player (simulated hand + deck + discard)
        #    We need to search discard too, in case an effect moved cards there unexpectedly
        #    or if the simulation incorrectly discarded something.
        current_sim_hand = list(player.hand)
        current_sim_deck = list(player.deck)
        current_sim_discard = list(player.discard_pile) # Include discard pile
        all_available_instances = current_sim_hand + current_sim_deck + current_sim_discard
        print(f"Searching within {len(all_available_instances)} total simulated instances (hand={len(current_sim_hand)}, deck={len(current_sim_deck)}, discard={len(current_sim_discard)}).")

        # 2. Find instances matching the actual hand names
        new_correct_hand_instances: List[Card] = []
        temp_available_instances = list(all_available_instances) # Work on a copy
        found_all_hand_cards = True
        not_found_hand_names = []

        target_hand_counts = {name: actual_hand_card_names.count(name) for name in set(actual_hand_card_names)}
        actual_hand_counts = {}

        for target_name, target_count in target_hand_counts.items():
            found_count_for_name = 0
            indices_to_remove = []
            for idx, instance in enumerate(temp_available_instances):
                if instance.name == target_name:
                    new_correct_hand_instances.append(instance)
                    indices_to_remove.append(idx)
                    found_count_for_name += 1
                    if found_count_for_name == target_count:
                        break # Found enough instances for this card name

            # Remove found instances from temp_available_instances (in reverse index order)
            for idx in sorted(indices_to_remove, reverse=True):
                temp_available_instances.pop(idx)

            actual_hand_counts[target_name] = found_count_for_name
            if found_count_for_name < target_count:
                found_all_hand_cards = False
                not_found_hand_names.append(f"{target_name} (found {found_count_for_name}/{target_count})")

        # 3. Validate hand synchronization
        if not found_all_hand_cards:
            print(f"ERROR: Could not find all required instances for the target hand!")
            print(f"Missing/Insufficient: {', '.join(not_found_hand_names)}")
            print(f"Current available instances searched: {[c.name for c in all_available_instances]}")
            # Decide on error handling: raise error, or proceed with partial sync?
            # For now, let's proceed but warn heavily. The state will be incorrect.
            print("WARNING: Proceeding with potentially incorrect hand state.")
            # Optionally raise ValueError("Synchronization failed: Could not find all hand cards.")

        # 4. Reconstruct the deck with remaining instances
        new_correct_deck_instances = temp_available_instances
        random.shuffle(new_correct_deck_instances) # Shuffle the inferred deck

        # 5. Validate deck size
        simulated_deck_size = len(new_correct_deck_instances)
        if simulated_deck_size != actual_deck_size:
            print(f"WARNING: Simulated deck size ({simulated_deck_size}) after sync does not match actual deck size ({actual_deck_size}).")
            print(f"  Simulated Deck Contents: {[c.name for c in new_correct_deck_instances]}")
            # This indicates a discrepancy between the total cards known to the simulation
            # and the total cards implied by the actual hand/deck state.
            # Possible causes:
            # - Incorrect input for actual hand/deck size.
            # - Cards lost/gained in the real game that the simulation doesn't know about.
            # - A fundamental bug in simulation's card tracking.
            # For now, we'll use the reconstructed deck, but the size mismatch is a problem.

        # 6. Update the player's actual hand and deck
        player.hand = new_correct_hand_instances
        player.deck = new_correct_deck_instances
        # Clear the discard pile, as the sync process accounts for all cards.
        # If a card truly belongs in discard according to the real game,
        # it wouldn't be in the actual_hand_card_names or implied deck.
        player.discard_pile = []

        print(f"Synchronization Complete for {player.name}:")
        print(f"  New Hand ({len(player.hand)} cards): {[c.name for c in player.hand]}")
        print(f"  New Deck Size: {len(player.deck)}")
        print("--- End Synchronization ---")
