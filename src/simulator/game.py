import random
from typing import List, Optional, Dict, Tuple, Any
from src.simulator.entities import Player, GameState, PokemonCard, Card, Attack, POINTS_TO_WIN # Changed to absolute import

# define the string constants for different actions
ACTION_PASS = "PASS"
ACTION_ATTACK_PREFIX = "ATTACK_"
ACTION_ATTACH_ENERGY_ACTIVE = "ATTACH_ENERGY_ACTIVE"
ACTION_ATTACH_ENERGY_BENCH_PREFIX = "ATTACH_ENERGY_BENCH_"
ACTION_PLAY_BASIC_BENCH_PREFIX = "PLAY_BASIC_BENCH_"

class Game:
    def __init__(self, player1_deck: List[Card], player2_deck: List[Card], player1_energy_types: List[str], player2_energy_types: List[str]):
        # make sure the decks are the correct size
        if len(player1_deck) != 20 or len(player2_deck) != 20:
            print(f"Decks should have 20 cards. P1: {len(player1_deck)}, P2: {len(player2_deck)}")

        self.player1 = Player("Player 1")
        self.player2 = Player("Player 2")
        # assign deck energy types
        self.player1.deck_energy_types = player1_energy_types if player1_energy_types else ["Colorless"]
        self.player2.deck_energy_types = player2_energy_types if player2_energy_types else ["Colorless"]
        # setup game for each player
        self.player1.setup_game(player1_deck[:20])
        self.player2.setup_game(player2_deck[:20])
        # setup the game state for the players
        self.game_state = GameState(self.player1, self.player2)
        self.turn_limit = 100 # prevent infinite loops in simulation
        self.actions_this_turn: Dict[str, Any] = {} # track actions

        print(f"--- Game Start ---")
        print(f"Player 1 Energy Types: {self.player1.deck_energy_types}")
        print(f"Player 2 Energy Types: {self.player2.deck_energy_types}")
        print(f"Starting Player: {self.game_state.get_current_player().name}")
        self._initialize_energy_stand() # sets up energy according to what the player picked
        self._start_turn()


    def _initialize_energy_stand(self):
        """Sets the initial state of the energy stand for both players."""
        self.player1.energy_stand_preview = random.choice(self.player1.deck_energy_types)
        self.player1.energy_stand_available = None

        self.player2.energy_stand_preview = random.choice(self.player2.deck_energy_types) 
        self.player2.energy_stand_available = None
        # the player actually going first will have their available stay none after _start_turn
        # the player going second will get their preview promoted in their first _start_turn

    def _start_turn(self):
        """Handles start-of-turn procedures including energy stand update."""
        player = self.game_state.get_current_player()
        print(f"\n--- Turn {self.game_state.turn_number}: {player.name}'s Turn ---")
        self.actions_this_turn = {"energy_attached": False} # reset turn actions

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

        player.draw_cards(1)

    def get_state_representation(self, player: Player) -> Dict[str, Any]:
        """
        Gather all relevant information about the current game state from the perspective of the
        given player, so that the AI can play/learn.
        """
        opponent = self.game_state.get_opponent() if player == self.game_state.get_current_player() else self.game_state.get_current_player()

        my_active_details = self._get_pokemon_details(player.active_pokemon)
        my_bench_details = [self._get_pokemon_details(p) for p in player.bench if p] 

        opp_active_details = self._get_pokemon_details(opponent.active_pokemon)

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
            "opp_bench_size": len(opponent.bench),

            # info about the game
            "turn": self.game_state.turn_number,
            "is_my_turn": player == self.game_state.get_current_player(),
            "can_attach_energy": not self.actions_this_turn.get("energy_attached", False),
            "is_first_turn": self.game_state.is_first_turn,
        }
        return state_dict

    def get_possible_actions(self) -> List[str]:
        """
        Get a list of valid actions for the current player.
        """
        player = self.game_state.get_current_player()
        opponent = self.game_state.get_opponent()
        actions = []

        # check the possible actions:
        can_attach_energy_this_turn = not self.actions_this_turn.get("energy_attached", False)
        is_starting_player_turn_1 = self.game_state.is_first_turn and \
                                    ((player == self.player1 and self.game_state.current_player_index == 0) or \
                                     (player == self.player2 and self.game_state.current_player_index == 1))

        # attach energy (person going first does not have energy available to attach, person going second does)
        if player.energy_stand_available and can_attach_energy_this_turn and not is_starting_player_turn_1:
             # check potential targets (active and bench)
             if player.active_pokemon and not player.active_pokemon.is_fainted:
                 actions.append(ACTION_ATTACH_ENERGY_ACTIVE)
             for i, bench_pokemon in enumerate(player.bench):
                 if bench_pokemon and not bench_pokemon.is_fainted:
                     actions.append(f"{ACTION_ATTACH_ENERGY_BENCH_PREFIX}{i}")

        # check attack actions
        if player.active_pokemon and not player.active_pokemon.is_fainted:
            for i, attack in enumerate(player.active_pokemon.attacks):
                 if player.active_pokemon.can_attack(i):
                     actions.append(f"{ACTION_ATTACK_PREFIX}{i}")

        # play a basic pokemon onto the bench
        if player.can_place_on_bench():
            for i, card in enumerate(player.hand):
                 # check if it's a pokemon card, and if it's basic
                 if isinstance(card, PokemonCard) and card.is_basic:
                     actions.append(f"{ACTION_PLAY_BASIC_BENCH_PREFIX}{i}")

        # TODO: Add actions for playing trainers, evolving, using abilities, retreating etc.

        # also have the option to just pass
        actions.append(ACTION_PASS)

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

        print(f"Player {player.name} attempts action: {action}")

        action_executed = False
        if action == ACTION_PASS:
            print("Turn passed.")
            action_executed = True # to make sure that we can pass without doing anything, which is valid

        # attach an energy to the active pokemon
        elif action == ACTION_ATTACH_ENERGY_ACTIVE:
            can_attach_energy_this_turn = not self.actions_this_turn.get("energy_attached", False)
            is_starting_player_turn_1 = self.game_state.is_first_turn and \
                                        ((player == self.player1 and self.game_state.current_player_index == 0) or \
                                         (player == self.player2 and self.game_state.current_player_index == 1))

            energy_to_attach = player.energy_stand_available # get the energy type from the stand

            if energy_to_attach and can_attach_energy_this_turn and not is_starting_player_turn_1 and player.active_pokemon:
                 player.active_pokemon.attach_energy(energy_to_attach)
                 player.energy_stand_available = None # actually consume the energy form the energy stand
                 self.actions_this_turn["energy_attached"] = True
                 action_executed = True
                 reward += 0.01 # i just decided to give a very small reward for using the energy, which in most cases you should do.
            else:
                 print(f"Cannot attach energy (Available: {energy_to_attach}, Attached: {not can_attach_energy_this_turn}, P1T1: {is_starting_player_turn_1}, No Active: {not player.active_pokemon}).")
                 if energy_to_attach and (not can_attach_energy_this_turn or is_starting_player_turn_1):
                     reward -= 0.1 # penalize for illegal move

        elif action.startswith(ACTION_ATTACK_PREFIX):
            try:
                # get which attack to use
                attack_index = int(action.split('_')[-1])
                if player.active_pokemon and opponent.active_pokemon:
                    if player.active_pokemon.can_attack(attack_index):
                        player.active_pokemon.attack(opponent.active_pokemon, attack_index)
                        action_executed = True

                        # check if opponents pokemon fainted
                        if opponent.active_pokemon.is_fainted:
                            points_scored = 2 if opponent.active_pokemon.is_ex else 1
                            print(f"{opponent.active_pokemon.name} fainted. {player.name} scores {points_scored} point(s).")
                            player.add_point(points_scored)
                            reward += 0.3 * points_scored # a decent reward for getting a KO

                            # we should check if the player won after getting a KO
                            winner = self.game_state.check_win_condition()
                            if winner:
                                print(f"Game Over. Winner: {winner.name}")
                                reward += 1.0 # give a big reward for winning
                                done = True
                            else:
                                # if the game is still going on, the opponent needs to mvoe a benched pokemon to active.
                                # check if the opponent has a benched pokemon
                                if not opponent.promote_bench_pokemon():
                                     print(f"Game Over. {opponent.name} has no Pokemon to promote. Winner: {player.name}")
                                     reward += 1.0 # reward for winning
                                     done = True
                    else:
                        print("Cannot attack (failed can_attack check).")
                        reward -= 0.1
                else:
                     print("Cannot attack (no active Pokemon for one or both players).")
                     reward -= 0.1
            except (ValueError, IndexError):
                print(f"Invalid attack action format: {action}")
                reward -= 0.1
        # option to play a basic pokemon
        elif action.startswith(ACTION_PLAY_BASIC_BENCH_PREFIX):
            try:
                # pick which card to play
                hand_index = int(action.split('_')[-1])
                if 0 <= hand_index < len(player.hand):
                    card_to_play = player.hand[hand_index]
                    if isinstance(card_to_play, PokemonCard) and card_to_play.is_basic:
                        if player.place_on_bench(card_to_play):
                             player.hand.pop(hand_index) # remove the card from hand
                             action_executed = True
                             reward += 0.05 # small reward for developing board
                        else:
                             print(f"Cannot play {card_to_play.name}: Bench is full.")
                             reward -= 0.1
                    else:
                         print(f"Cannot play card at index {hand_index}: Not a Basic Pokemon.")
                         reward -= 0.1
                else:
                     print(f"Invalid hand index in action: {action}")
                     reward -= 0.1
            except (ValueError, IndexError):
                 print(f"Invalid play basic action format: {action}")
                 reward -= 0.1
        
        # attaching an energy to a benched pokemon
        elif action.startswith(ACTION_ATTACH_ENERGY_BENCH_PREFIX):
            can_attach_energy_this_turn = not self.actions_this_turn.get("energy_attached", False)
            is_starting_player_turn_1 = self.game_state.is_first_turn and \
                                        ((player == self.player1 and self.game_state.current_player_index == 0) or \
                                         (player == self.player2 and self.game_state.current_player_index == 1))
            energy_to_attach = player.energy_stand_available

            if energy_to_attach and can_attach_energy_this_turn and not is_starting_player_turn_1:
                try:
                    bench_index = int(action.split('_')[-1])
                    if 0 <= bench_index < len(player.bench):
                        target_pokemon = player.bench[bench_index]
                        if target_pokemon and not target_pokemon.is_fainted:
                             target_pokemon.attach_energy(energy_to_attach)
                             player.energy_stand_available = None # Consume energy (Renamed)
                             self.actions_this_turn["energy_attached"] = True
                             action_executed = True # Attaching energy doesn't end turn
                             reward += 0.01
                        else:
                             print(f"Cannot attach energy to bench {bench_index}: Pokemon fainted or invalid.")
                             reward -= 0.1
                    else:
                         print(f"Invalid bench index in action: {action}")
                         reward -= 0.1
                except (ValueError, IndexError):
                     print(f"Invalid attach energy bench action format: {action}")
                     reward -= 0.1
            else:
                 print(f"Cannot attach energy (Available: {energy_to_attach}, Attached: {not can_attach_energy_this_turn}, P1T1: {is_starting_player_turn_1}).") # Renamed
                 if energy_to_attach and (not can_attach_energy_this_turn or is_starting_player_turn_1):
                     reward -= 0.1


        else:
            print(f"Unknown or invalid action attempted: {action}")
            reward = -0.1

        # after the action is done: 
        if done:
            next_state = self.get_state_representation(player)
            return next_state, reward, done

        # if action was an attack or pass, end the turn
        if action.startswith(ACTION_ATTACK_PREFIX) or action == ACTION_PASS:
            print("--- Pokemon Checkup ---")
            # TODO: need to implement status effects (poison could kill pokemon during this phase)
            self.game_state.switch_turn()
            # check the turn limit
            if self.game_state.turn_number > self.turn_limit:
                print("Game Over! Turn limit reached.")
                done = True
                reward = -0.5 # i put a penalty for reaching the draw limit, because I don't want the ai to just be passing or making inefficient plays
            else:
                 self._start_turn()

        next_player = self.game_state.get_current_player()
        next_state = self.get_state_representation(next_player)

        return next_state, reward, done

    def run_simple_test_game(self):
        """Runs a very simple game with random actions for testing."""
        print("\n--- Starting Simple Test Game (Random Actions) ---")
        for _ in range(self.turn_limit):
            current_player = self.game_state.get_current_player()
            print(f"\nPlayer: {current_player.name}")
            print(f"Hand: {[c.name for c in current_player.hand]}")
            print(f"Active: {current_player.active_pokemon}")
            print(f"Bench: {current_player.bench}")

            possible_actions = self.get_possible_actions()
            print(f"Possible Actions: {possible_actions}")

            if not possible_actions:
                 print("No possible actions! Something is wrong.")
                 break
            
            chosen_action = random.choice(possible_actions)
            print(f"Chosen Action: {chosen_action}")

            _, reward, done = self.step(chosen_action)
            print(f"Reward: {reward}, Done: {done}")

            if done:
                print("--- Test Game Finished ---")
                break
        else:
             print("--- Test Game Finished (Turn Limit Reached) ---")


if __name__ == "__main__":
    # example attacks
    vine_whip = Attack(name="Vine Whip", cost={"Grass": 1, "Colorless": 1}, damage=40)
    scratch = Attack(name="Scratch", cost={"Fire": 1}, damage=20)
    water_gun = Attack(name="Water Gun", cost={"Water": 1}, damage=20)
    gnaw = Attack(name="Gnaw", cost={"Electric": 1}, damage=20)

    # pokemon with name, hp, attacks, type, and weaknesses
    bulbasaur = PokemonCard("Bulbasaur", 70, [vine_whip], "Grass", "Fire", False, True)
    slandit = PokemonCard("Slandit", 60, [scratch], "Fire", "Water", False, True)
    squirtle = PokemonCard("Squirtle", 60, [water_gun], "Water", "Electric", False, True)
    pikachu = PokemonCard("Pikachu", 60, [gnaw], "Electric", "Fighting", False, True)
