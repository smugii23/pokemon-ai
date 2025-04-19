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

class TrainerCard(Card):
    """Represents a Trainer card (Supporter, Item, Tool, Stadium)."""
    def __init__(self, name: str, trainer_type: str, effect_tag: Optional[str] = None):
        super().__init__(name)
        self.trainer_type = trainer_type # e.g., "Supporter", "Item", "Tool"
        self.effect_tag = effect_tag

    def __repr__(self):
        return f"TrainerCard({self.name}, Type: {self.trainer_type}, Effect: {self.effect_tag})"

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
                 retreat_cost: int = 0, # Added retreat cost
                 is_ex: bool = False, is_basic: bool = True, ability: Optional[Dict] = None): # Added ability
        super().__init__(name)
        self.hp = hp
        self.current_hp = hp
        self.retreat_cost = retreat_cost # Store retreat cost
        self.attacks = attacks
        self.pokemon_type = pokemon_type
        self.weakness_type = weakness_type
        self.is_ex = is_ex
        self.is_basic = is_basic
        self.ability = ability # Store ability data
        self.attached_energy: Dict[str, int] = {}
        self.attached_tool: Optional[TrainerCard] = None # Added attribute for tool
        self.is_active = False
        self.is_fainted = False

    def can_retreat(self, current_turn_cost_modifier: int = 0) -> bool:
        """Check if the Pokemon has enough energy to retreat."""
        if self.is_fainted:
            return False
        effective_cost = max(0, self.retreat_cost + current_turn_cost_modifier)
        total_attached = sum(self.attached_energy.values())
        return total_attached >= effective_cost

    def can_attack(self, attack_index: int) -> bool:
        """Check if the Pokemon has enough energy for a specific attack."""
        if self.is_fainted: return False
        if not (0 <= attack_index < len(self.attacks)): return False

        attack = self.attacks[attack_index]
        attack_cost = attack.cost
        total_cost = sum(attack_cost.values())
        total_attached = sum(self.attached_energy.values())

        if total_attached < total_cost: return False

        available_specific_energy = self.attached_energy.copy()
        required_specific_cost = {k: v for k, v in attack_cost.items() if k != "Colorless"}

        for energy_type, cost in required_specific_cost.items():
            if available_specific_energy.get(energy_type, 0) < cost: return False
            available_specific_energy[energy_type] -= cost

        remaining_attached = sum(available_specific_energy.values())
        colorless_cost = attack_cost.get("Colorless", 0)
        return remaining_attached >= colorless_cost

    # --- NEW perform_attack METHOD ---
    def perform_attack(self, attack_index: int, target: 'PokemonCard', damage_modifier: int = 0) -> int:
        """
        Performs the attack calculation, applies damage to the target,
        handles weakness/resistance, and returns the damage dealt.
        Called by Game.step().
        """
        if not self.can_attack(attack_index):
            print(f"Error: {self.name} cannot use attack {attack_index} (checked in perform_attack).")
            return 0 # Return 0 damage if cannot attack

        chosen_attack = self.attacks[attack_index]
        calculated_damage = chosen_attack.damage

        print(f"{self.name} ({self.pokemon_type}) uses {chosen_attack.name} on {target.name} ({target.pokemon_type}).")
        print(f"  Base damage: {chosen_attack.damage}")

        # Apply direct damage modifiers (e.g., from Red)
        if damage_modifier > 0:
            calculated_damage += damage_modifier
            print(f"  Applying modifier: +{damage_modifier} damage (Total: {calculated_damage})")

        # Check target's weakness
        if target.weakness_type == self.pokemon_type:
            calculated_damage += 20 # TCG Pocket weakness bonus
            print(f"  It's super effective! +20 weakness damage (Total: {calculated_damage})")

        # TODO: Check target's resistance (if implemented in card data)
        # if target.resistance_type == self.pokemon_type:
        #    calculated_damage = max(0, calculated_damage - resistance_amount)
        #    print(f"  It's not very effective! -{resistance_amount} resistance damage (Total: {calculated_damage})")

        # Apply damage to the target, passing self as the attacker
        target.take_damage(calculated_damage, self)

        # TODO: Implement attack effects based on chosen_attack.effect
        # e.g., if chosen_attack.effect == "DISCARD_ENERGY_SELF": self.discard_energy(1)
        # e.g., if chosen_attack.effect == "PARALYZE_TARGET": target.apply_status("paralyzed")

        return calculated_damage # Return the final damage dealt for reward calculation

    # --- MODIFIED take_damage METHOD ---
    def take_damage(self, damage: int, attacker: Optional['PokemonCard'] = None):
        """
        Apply damage to the Pokemon. Now accepts the attacker for effects like Rocky Helmet.
        """
        if self.is_fainted: return

        # Ensure damage is non-negative before applying
        actual_damage = max(0, damage)
        self.current_hp -= actual_damage
        print(f"  {self.name} took {actual_damage} damage, remaining HP: {max(0, self.current_hp)}")

        # --- Apply Tool Effects (e.g., Rocky Helmet) triggered by taking damage ---
        if self.attached_tool and self.attached_tool.effect_tag == "TRAINER_TOOL_ROCKY_HELMET_IF_THE_POKÉMON_THIS_CARD": # Corrected tag check
            if attacker and not attacker.is_fainted: # Check if attacker exists and isn't already fainted
                print(f"  Rocky Helmet on {self.name} triggers!")
                # Apply damage back to the attacker. Note: This damage application
                # itself won't trigger another Rocky Helmet (no infinite loops).
                attacker.take_damage(20) # Attacker takes 20 damage
            elif attacker:
                 print(f"  Rocky Helmet on {self.name} triggers, but attacker {attacker.name} is already fainted.")
            else:
                 print(f"  Rocky Helmet on {self.name} triggers, but attacker information is missing.")


        # --- Check for Faint ---
        if self.current_hp <= 0:
            print(f"{self.name} has fainted!")
            self.is_fainted = True
            self.current_hp = 0
            self.attached_energy = {} # Energy is discarded on faint

            # Tool is discarded when Pokemon faints
            if self.attached_tool:
                print(f"  Discarding attached tool: {self.attached_tool.name}")
                # TODO: Add the tool to the correct Player's discard pile (needs Player context)
                self.attached_tool = None
            # TODO: Handle other effects clearing on faint (status conditions etc.)


    def attach_energy(self, energy_type: str):
        """Attach an energy of a specific type."""
        if self.is_fainted: return
        self.attached_energy[energy_type] = self.attached_energy.get(energy_type, 0) + 1
        print(f"Attached {energy_type} energy to {self.name}. Total: {self.attached_energy}")

    def discard_energy(self, amount: int):
        """Discards a specified amount of energy randomly."""
        if self.is_fainted: return
        print(f"Attempting to discard {amount} energy from {self.name}...")
        discarded_count = 0
        energy_pool = []
        for energy_type, count in self.attached_energy.items():
            energy_pool.extend([energy_type] * count)

        random.shuffle(energy_pool)

        while discarded_count < amount and energy_pool:
            energy_to_discard = energy_pool.pop()
            self.attached_energy[energy_to_discard] -= 1
            if self.attached_energy[energy_to_discard] == 0:
                del self.attached_energy[energy_to_discard]
            discarded_count += 1
            print(f"  Discarded 1 {energy_to_discard} energy.")

        if discarded_count < amount:
            print(f"  Warning: Could only discard {discarded_count} energy (required {amount}).")
        print(f"  {self.name} remaining energy: {self.attached_energy}")


    def __repr__(self):
        energy_str = ", ".join(f"{t}:{c}" for t, c in self.attached_energy.items())
        attack_names = ", ".join(a.name for a in self.attacks)
        status = " (Active)" if self.is_active else ""
        status += " (Fainted)" if self.is_fainted else ""
        ability_str = f" Ability:{self.ability['name']}" if self.ability else ""
        tool_str = f" Tool:{self.attached_tool.name}" if self.attached_tool else "" # Added tool display
        retreat_str = f" Retreat:{self.retreat_cost}" # Added retreat cost display
        return (f"{self.name} [{self.pokemon_type}] "
                f"HP:{self.current_hp}/{self.hp}{retreat_str} " # Added retreat_str
                f"Attacks:[{attack_names}]{ability_str}{tool_str} "
                f"NRG:[{energy_str}]{status}")



class Player:
    """Represents a player's state."""
    def __init__(self, name: str):
        self.name = name
        self.deck: List[Card] = []
        self.hand: List[Card] = []
        self.discard_pile: List[Card] = []
        self.points: int = 0
        self.deck_energy_types: List[str] = []
        self.energy_stand_available: Optional[str] = None # energy available to attach this turn
        self.energy_stand_preview: Optional[str] = None # energy available next turn
        self.active_pokemon: Optional[PokemonCard] = None
        self.bench: List[PokemonCard] = []
        # Attributes for simultaneous setup phase
        self.pending_setup_active: Optional[PokemonCard] = None
        self.pending_setup_bench: List[PokemonCard] = []
        self.setup_ready: bool = False
        self.skip_automatic_draw: bool = False

    def setup_game(self, deck: List[Card]):
        """Initial game setup, ensuring a Basic Pokemon is in the opening hand."""
        self.deck = list(deck) # Keep the original passed deck structure
        self.hand = []
        self.discard_pile = []
        self.points = 0
        self.active_pokemon = None
        self.bench = []
        self.pending_setup_active = None
        self.pending_setup_bench = []
        self.setup_ready = False

        # --- Enforce Guaranteed Basic Rule ---
        found_valid_hand = False
        attempts = 0
        max_attempts = 100 # Safety break for potential infinite loops if deck has no basics
        original_deck_state = list(self.deck) # Keep original deck state for resets

        while not found_valid_hand and attempts < max_attempts:
            attempts += 1
            # Reset deck and hand for redraw attempt
            self.deck = list(original_deck_state) # Use the original full deck
            random.shuffle(self.deck)
            self.hand = [] # Clear hand before drawing

            # Draw the initial hand using the helper method
            self._draw_initial_hand() # Draws STARTING_HAND_SIZE cards into self.hand

            # Check if this hand contains a Basic Pokemon
            has_basic = any(isinstance(card, PokemonCard) and card.is_basic for card in self.hand)

            if has_basic:
                found_valid_hand = True
                if attempts > 1:
                    print(f"{self.name} found a Basic Pokemon after {attempts} draw attempts.")
            # else: # Loop continues if no basic found
                # print(f"Attempt {attempts}: No Basic Pokemon found in hand for {self.name}. Redrawing...") # Optional debug

        if not found_valid_hand:
            # This should ideally never happen if the deck contains basics
            raise RuntimeError(f"Setup failed: Could not draw a hand with a Basic Pokemon for {self.name} after {max_attempts} attempts. Check deck composition.")
        # --- End Guaranteed Basic Enforcement ---

        # --- Proceed with setup using the valid hand ---
        # Find the first Basic Pokemon in the now-guaranteed valid hand
        active_index = -1
        for i, card in enumerate(self.hand):
            if isinstance(card, PokemonCard) and card.is_basic:
                active_index = i
                break

        # This check should now be redundant due to the loop above, but keep as safeguard
        if active_index == -1:
            raise RuntimeError(f"Internal Setup Error: No Basic Pokemon found in hand for {self.name} despite guarantee logic.")

        # --- Active Pokemon Selection and Benching Moved ---
        # The selection of the active Pokemon and placement of initial bench Pokemon
        # will now be handled by the Game class (via rl_env/play_vs_ai) after the initial hands are drawn
        # and validated by this setup_game method.
        # DO NOT set self.active_pokemon here.
        print(f"{self.name} setup complete. Hand drawn and validated (contains basic).")
        print(f"{self.name} starting hand size: {len(self.hand)}")
        # Method now implicitly returns None, indicating setup is done for this player.


    def _draw_initial_hand(self):
        """Helper to draw the initial hand, ensuring deck isn't empty."""
        draw_count = min(STARTING_HAND_SIZE, len(self.deck))
        if draw_count < STARTING_HAND_SIZE:
             print(f"Warning: Deck has only {draw_count} cards for initial draw.")
        self.hand.extend(self.deck[:draw_count])
        self.deck = self.deck[draw_count:]
        print(f"{self.name} drew initial hand of {len(self.hand)} cards.")


    def draw_cards(self, num: int):
        """Draw cards from the deck during the game, respecting max hand size."""
        drawn_count = 0
        for _ in range(num):
            if not self.deck:
                print(f"{self.name} cannot draw, deck is empty!")
                break
            if len(self.hand) >= MAX_HAND_SIZE:
                print(f"{self.name} cannot draw, hand is full (Max {MAX_HAND_SIZE})!")
                # --- MODIFICATION: Discard if hand is full ---
                # In the actual game, if the hand is full, the drawn card is shown and discarded.
                card = self.deck.pop(0)
                self.discard_pile.append(card)
                print(f"  (Hand full, discarded drawn card: {card.name})")
                # Do not increment drawn_count as it didn't go to hand
                break # Stop drawing if hand is full

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
        energy_stand_str = f"Available:{self.energy_stand_available or 'None'}, Preview:{self.energy_stand_preview or 'None'}" # Updated names
        return (f"Player({self.name}, Pts:{self.points}, Hand:{len(self.hand)}, Deck:{len(self.deck)}, "
                f"Discard:{len(self.discard_pile)}, EnergyStand:[{energy_stand_str}], "
                f"Active:{active_str}, Bench:[{bench_str}])")


class GameState:
    """Encapsulates the entire game state."""
    def __init__(self, player1: Player, player2: Player):
        self.players = [player1, player2]
        # Determine and store the starting player index from the "coin toss"
        self.starting_player_index = random.choice([0, 1])
        self.current_player_index = self.starting_player_index # Start with the designated player
        self.turn_number = 1
        self.is_first_turn = True # Flag for the very first turn rules

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
