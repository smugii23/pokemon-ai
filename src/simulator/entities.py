import random
from typing import List, Optional, Dict, Tuple

MAX_BENCH_SIZE = 3
MAX_HAND_SIZE = 10
STARTING_HAND_SIZE = 5
POINTS_TO_WIN = 3

class Card:
    """Base class for all cards."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

class Attack:
    """Represents a single attack with its properties."""
    def __init__(self, name: str, cost: Dict[str, int], damage: int, effect=None):
        self.name = name
        self.cost = cost
        self.damage = damage
        self.effect = effect

    def __repr__(self):
        cost_str = ", ".join(f"{t}:{c}" for t, c in self.cost.items())
        return f"Attack(Name: {self.name}, Cost: [{cost_str}], Dmg: {self.damage})"

class PokemonCard(Card):
    """A class for cards that are pokemon"""
    def __init__(self, name: str, hp: int, attacks: List[Attack],
                 pokemon_type: str = "Colorless", weakness_type: Optional[str] = None,
                 is_ex: bool = False, is_basic: bool = True): # Added is_basic flag
        super().__init__(name)
        self.hp = hp
        self.current_hp = hp
        # decided to create a separate class for attacks to hold multiple attacks
        self.attacks = attacks
        self.pokemon_type = pokemon_type
        self.weakness_type = weakness_type
        self.is_ex = is_ex
        self.is_basic = is_basic # Store the flag
        self.attached_energy: Dict[str, int] = {}
        self.is_active = False
        self.is_fainted = False

    def can_attack(self, attack_index: int) -> bool:
        """Check if the Pokemon has enough energy for a specific attack."""
        if self.is_fainted:
            return False
        if not (0 <= attack_index < len(self.attacks)):
            # print(f"Error: Invalid attack index {attack_index} for {self.name}") # Optional: too verbose for AI?
            return False

        attack = self.attacks[attack_index]
        attack_cost = attack.cost
        total_cost = sum(attack_cost.values())
        total_attached = sum(self.attached_energy.values())

        # check total energy first
        if total_attached < total_cost:
            return False

        # check specific energy types
        available_specific_energy = self.attached_energy.copy()
        required_specific_cost = {k: v for k, v in attack_cost.items() if k != "Colorless"}

        for energy_type, cost in required_specific_cost.items():
            if available_specific_energy.get(energy_type, 0) < cost:
                return False
            available_specific_energy[energy_type] -= cost # Consume specific energy for check

        # check remaining attached energy against colorless cost
        remaining_attached = sum(available_specific_energy.values())
        colorless_cost = attack_cost.get("Colorless", 0)
        return remaining_attached >= colorless_cost

    def attack(self, target: 'PokemonCard', attack_index: int):
        """Perform the specified attack, calculating weakness."""
        if not (0 <= attack_index < len(self.attacks)):
            print(f"Error: Invalid attack index {attack_index} for {self.name}")
            return

        chosen_attack = self.attacks[attack_index]

        if self.can_attack(attack_index):
            damage = chosen_attack.damage
            print(f"{self.name} ({self.pokemon_type}) uses {chosen_attack.name} on {target.name} ({target.pokemon_type}). Base damage: {damage}")

            # check weakness (+20 damage in tcg pocket)
            if target.weakness_type == self.pokemon_type:
                damage += 20
                print(f"It's super effective! (+20 weakness damage)")

            # TODO: implement attack effects

            target.take_damage(damage, attacker_type=self.pokemon_type)
        else:
            print(f"{self.name} cannot use {chosen_attack.name} yet.")

    def take_damage(self, damage: int, attacker_type: Optional[str] = None):
        """Apply damage to the Pokemon."""
        if self.is_fainted: return

        self.current_hp -= damage
        print(f"{self.name} took {damage} damage, remaining HP: {max(0, self.current_hp)}")
        if self.current_hp <= 0:
            print(f"{self.name} has fainted!")
            self.is_fainted = True
            self.current_hp = 0
            self.attached_energy = {}

    def attach_energy(self, energy_type: str):
        """Attach an energy of a specific type."""
        if self.is_fainted: return
        self.attached_energy[energy_type] = self.attached_energy.get(energy_type, 0) + 1
        print(f"Attached {energy_type} energy to {self.name}. Total: {self.attached_energy}")

    def __repr__(self):
        energy_str = ", ".join(f"{t}:{c}" for t, c in self.attached_energy.items())
        attack_names = ", ".join(a.name for a in self.attacks)
        status = " (Active)" if self.is_active else ""
        status += " (Fainted)" if self.is_fainted else ""
        return (f"{self.name} [{self.pokemon_type}] "
                f"HP:{self.current_hp}/{self.hp} "
                f"Attacks:[{attack_names}] "
                f"NRG:[{energy_str}]{status}")


class Player:
    """Represents a player's state."""
    def __init__(self, name: str):
        self.name = name
        self.deck: List[Card] = []
        self.hand: List[Card] = []
        self.discard_pile: List[Card] = []
        self.points: int = 0
        # self.energy_zone: Dict[str, int] = {"Colorless": 0} # Replaced by energy stand logic
        self.deck_energy_types: List[str] = [] # e.g., ["Fire", "Colorless"] - Set during game setup
        self.energy_stand_outer: Optional[str] = None # Energy available to attach this turn
        self.energy_stand_inner: Optional[str] = None # Energy available next turn
        self.active_pokemon: Optional[PokemonCard] = None
        self.bench: List[PokemonCard] = []

    def setup_game(self, deck: List[Card]):
        """Initial game setup."""
        self.deck = list(deck)
        random.shuffle(self.deck)
        self.draw_cards(STARTING_HAND_SIZE)
        active_found = False
        for i in range(len(self.hand) -1, -1, -1):
             card = self.hand[i]
             # TODO: Add check for "Basic" Pokemon when that concept is added
             if isinstance(card, PokemonCard):
                 self.active_pokemon = self.hand.pop(i)
                 self.active_pokemon.is_active = True
                 print(f"{self.name} sets {self.active_pokemon.name} as active.")
                 active_found = True
                 break
        # No need for 'if not active_found' check, as basic is guaranteed in TCG Pocket

    def draw_cards(self, num: int):
        """Draw cards from the deck, respecting max hand size."""
        drawn_count = 0
        for _ in range(num):
            if not self.deck:
                print(f"{self.name} cannot draw, deck is empty!")
                break
            if len(self.hand) >= MAX_HAND_SIZE:
                print(f"{self.name} cannot draw, hand is full (Max {MAX_HAND_SIZE})!")
                break

            # draw a card from the top
            card = self.deck.pop(0)
            self.hand.append(card)
            drawn_count += 1

        if drawn_count > 0:
            print(f"{self.name} drew {drawn_count} card(s). Hand size: {len(self.hand)}")

    def add_point(self, count: int = 1):
        """Add points to the player's score."""
        self.points += count
        print(f"{self.name} scores {count} point(s)! Total points: {self.points}")

    def can_place_on_bench(self) -> bool:
        """Check if there is space on the bench."""
        return len(self.bench) < MAX_BENCH_SIZE

    def place_on_bench(self, pokemon: PokemonCard):
        """Place a Pokemon on the bench if possible."""
        if self.can_place_on_bench():
            self.bench.append(pokemon)
            pokemon.is_active = False
            print(f"{self.name} placed {pokemon.name} on the bench.")
            return True
        else:
            print(f"{self.name}'s bench is full (Max {MAX_BENCH_SIZE}). Cannot place {pokemon.name}.")
            return False

    def promote_bench_pokemon(self) -> bool:
        """Promote the first available bench Pokemon to active if active is fainted."""
        if self.active_pokemon and not self.active_pokemon.is_fainted:
            print("Cannot promote, active Pokemon is not fainted.")
            return False
        if not self.bench:
            print("Cannot promote, bench is empty.")
            return False
        new_active = self.bench.pop(0)
        # put the fainted pokemon in discard pile
        if self.active_pokemon:
             self.discard_pile.append(self.active_pokemon)
             print(f"Discarded fainted {self.active_pokemon.name}.")

        self.active_pokemon = new_active
        self.active_pokemon.is_active = True
        print(f"{self.name} promoted {self.active_pokemon.name} from the bench to active.")
        return True


    def __repr__(self):
        active_str = self.active_pokemon.name if self.active_pokemon else 'None'
        bench_str = ", ".join(p.name for p in self.bench)
        energy_stand_str = f"Outer:{self.energy_stand_outer or 'None'}, Inner:{self.energy_stand_inner or 'None'}"
        return (f"Player({self.name}, Pts:{self.points}, Hand:{len(self.hand)}, Deck:{len(self.deck)}, "
                f"Discard:{len(self.discard_pile)}, EnergyStand:[{energy_stand_str}], "
                f"Active:{active_str}, Bench:[{bench_str}])")


class GameState:
    """Encapsulates the entire game state."""
    def __init__(self, player1: Player, player2: Player):
        self.players = [player1, player2]
        self.current_player_index = random.choice([0, 1])
        self.turn_number = 1
        self.is_first_turn = True

    def get_current_player(self) -> Player:
        return self.players[self.current_player_index]

    def get_opponent(self) -> Player:
        return self.players[1 - self.current_player_index]

    def switch_turn(self):
        """Pass the turn to the other player and increment turn counter."""
        self.current_player_index = 1 - self.current_player_index
        if self.current_player_index == 0:
             self.turn_number += 1
        self.is_first_turn = False
        print(f"\n--- Turn {self.turn_number}: {self.get_current_player().name}'s Turn ---")

    def check_win_condition(self) -> Optional[Player]:
        """Check TCG Pocket win conditions."""
        # 1. player reaches 3 points
        for player in self.players:
            if player.points >= POINTS_TO_WIN:
                print(f"{player.name} has reached {player.points} points!")
                return player

        # 2. opponent has no pokemon in play after active faints
        opponent = self.get_opponent()
        if opponent.active_pokemon and opponent.active_pokemon.is_fainted and not opponent.bench:
             print(f"{opponent.name} has no Pokemon left to promote!")
             return self.get_current_player()


    def __repr__(self):
        return (f"GameState(Turn: {self.turn_number} - {self.get_current_player().name}, "
                f"P1: {self.players[0]}, P2: {self.players[1]})")


# example game
if __name__ == "__main__":
    # example attacks (i made some up)
    vine_whip = Attack(name="Vine Whip", cost={"Grass": 1, "Colorless": 1}, damage=40)
    scratch = Attack(name="Scratch", cost={"Fire": 1}, damage=20)
    water_gun = Attack(name="Water Gun", cost={"Water": 1}, damage=20)
    gnaw = Attack(name="Gnaw", cost={"Electric": 1}, damage=20)

    # pokemon with name, hp, attacks, type, and weaknesses
    bulba = PokemonCard("Bulbasaur", 70, [vine_whip], "Grass", "Fire", False, True)
    slandit = PokemonCard("Slandit", 60, [scratch], "Fire", "Water", False, True)
    squirtle = PokemonCard("Squirtle", 60, [water_gun], "Water", "Electric", False, True)
    pikachu = PokemonCard("Pikachu", 60, [gnaw], "Electric", "Fighting", False, True)

    # for now, i'll just create simple decks that are illegal (can only have 2 of each card)
    deck1_cards = [pikachu] * 6 + [bulba] * 6 + [slandit] * 8
    deck2_cards = [bulba] * 6 + [pikachu] * 6 + [slandit] * 8
    random.shuffle(deck1_cards)
    random.shuffle(deck2_cards)

    # create players
    player1 = Player("Shuma")
    player2 = Player("Ash")

    # setup the game
    player1.setup_game(deck1_cards[:20])
    player2.setup_game(deck2_cards[:20])

    # create the game state
    game = GameState(player1, player2)
    print("--- Initial Game State ---")
    print(game)
    print(f"Starting Player: {game.get_current_player().name}")

    winner = game.check_win_condition()
    if winner:
        print(f"\nGame Over! Winner: {winner.name}")
    else:
        print("\n--- End of Example ---")
        print(game)
