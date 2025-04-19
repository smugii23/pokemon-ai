import json
import re
import os
import traceback
from typing import Dict, List, Optional, Any

# --- Constants ---
INPUT_FILE = os.path.join(os.path.dirname(__file__), 'raw_card_data.txt')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'cards.json')

# --- Mappings ---
# Assuming these symbols based on common usage and examples
ENERGY_SYMBOL_MAP: Dict[str, str] = {
    'G': 'Grass', 'R': 'Fire', 'W': 'Water', 'L': 'Electric',
    'P': 'Psychic', 'F': 'Fighting', 'D': 'Darkness', 'M': 'Metal',
    'Y': 'Fairy', # Assuming Y for Fairy
    'C': 'Colorless', 'N': 'Dragon' # Assuming N for Dragon
}

# --- Helper Functions ---

def parse_energy_cost(cost_str: str) -> Dict[str, int]:
    """Parses an energy cost string (e.g., '[P][P][P][C]' or 'PPPC') into a dictionary."""
    cost_dict: Dict[str, int] = {}
    if not cost_str or not isinstance(cost_str, str):
        return cost_dict

    # Find all occurrences of bracketed symbols (e.g., [P]) or single characters
    symbols = re.findall(r'\[(.)\]|(.)', cost_str) # Matches [X] or Y

    for bracketed, single in symbols:
        symbol = bracketed if bracketed else single # Use the captured group
        energy_type = ENERGY_SYMBOL_MAP.get(symbol)
        if energy_type:
            cost_dict[energy_type] = cost_dict.get(energy_type, 0) + 1
        else:
            print(f"Warning: Unknown energy symbol '{symbol}' found in cost string '{cost_str}'")
    return cost_dict

def generate_effect_tag(prefix: str, name: str, text: Optional[str]) -> Optional[str]:
    """Generates a placeholder effect tag from text."""
    if not text:
        return None
    # Basic normalization: uppercase, replace non-alphanumeric with underscore
    normalized_text = re.sub(r'\W+', '_', text).strip('_').upper()
    normalized_name = re.sub(r'\W+', '_', name).strip('_').upper()
    # Limit length to avoid excessively long tags
    max_len = 50
    tag = f"{prefix}_{normalized_name}_{normalized_text}"[:max_len]
    return tag if tag else None

# --- Main Parsing Logic ---

def parse_raw_data(input_filepath: str = INPUT_FILE, output_filepath: str = OUTPUT_FILE):
    """Reads raw card text data, parses it, and writes to JSON."""
    all_cards_data: List[Dict[str, Any]] = []
    print(f"Starting parsing process for {input_filepath}...")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            # Split by at least three newlines (two blank lines), filter empty strings
            content = f.read()
            # Use a regex that matches 3 or more newlines, allowing whitespace on blank lines
            card_blocks = [block.strip() for block in re.split(r'\n\s*\n\s*\n+', content) if block.strip()]
        print(f"Found {len(card_blocks)} potential card blocks.")

        for i, block in enumerate(card_blocks):
            print(f"\n--- Processing Block {i+1} ---")
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines:
                print("Skipping empty block.")
                continue

            card_data: Dict[str, Any] = { # Initialize with defaults
                "name": None,
                "card_type": None,
                "hp": None,
                "pokemon_type": None,
                "evolves_from": None,
                "is_basic": None,
                "is_ex": False,
                "weakness_type": None,
                "retreat_cost": None,
                "attacks": [],
                "ability": None,
                "effect_text": None, # For Trainers/Energy
                "effect_tag": None   # For Trainers/Energy
            }
            current_line_index = 0
            try:
                # --- Basic Info ---
                card_data['name'] = lines[current_line_index]
                print(f"  Name: {card_data['name']}")
                card_data['is_ex'] = " ex" in card_data['name'] # Check if name contains " ex"
                current_line_index += 1

                # Type/HP (Pokemon only)
                if lines[current_line_index].startswith('-'):
                    match = re.match(r'-\s*(.*?)\s*-\s*(\d+)\s*HP', lines[current_line_index])
                    if match:
                        card_data['pokemon_type'] = match.group(1).strip()
                        card_data['hp'] = int(match.group(2))
                        print(f"  Pokemon Type: {card_data['pokemon_type']}, HP: {card_data['hp']}")
                    else:
                         print(f"  Warning: Could not parse Type/HP line: {lines[current_line_index]}")
                    current_line_index += 1

                # Card Type / Stage
                card_type_line = lines[current_line_index]
                print(f"  Type Line: {card_type_line}")
                if "Pokémon" in card_type_line:
                    card_data['card_type'] = "Pokemon"
                    card_data['is_basic'] = "Basic" in card_type_line
                    if "Stage 1" in card_type_line:
                         # Attempt to extract evolves_from, might need refinement
                         match = re.search(r'Evolves from (.*)', card_type_line)
                         card_data['evolves_from'] = match.group(1).strip() if match else None
                         print(f"  Stage 1, Evolves from: {card_data['evolves_from']}")
                    elif "Stage 2" in card_type_line:
                         match = re.search(r'Evolves from (.*)', card_type_line)
                         card_data['evolves_from'] = match.group(1).strip() if match else None
                         print(f"  Stage 2, Evolves from: {card_data['evolves_from']}")
                    else:
                        print(f"  Basic Pokemon")

                elif "Trainer" in card_type_line:
                    type_match = re.search(r'Trainer\s*-\s*(.*)', card_type_line)
                    card_data['card_type'] = type_match.group(1).strip() if type_match else "Trainer" # Supporter, Item, Tool, Stadium
                    print(f"  Trainer Type: {card_data['card_type']}")
                elif "Energy" in card_type_line:
                    card_data['card_type'] = "Energy" # Basic or Special?
                    print(f"  Energy Card")
                else:
                    print(f"  Warning: Could not determine card type from line: {card_type_line}")
                current_line_index += 1

                # --- Attacks / Abilities / Trainer Effects ---
                active_attack: Optional[Dict] = None # Track attack being defined

                while current_line_index < len(lines) and \
                      not lines[current_line_index].startswith("Weakness:") and \
                      not lines[current_line_index].startswith("Retreat:") and \
                      not lines[current_line_index].startswith("ex rule:"):

                    line = lines[current_line_index]
                    print(f"  Processing line: {line}")

                    # Check for Ability Start
                    if line.startswith("Ability:"):
                        active_attack = None # Finish any previous attack
                        ability_name = line.split(":", 1)[1].strip()
                        ability_text_lines = []
                        current_line_index += 1 # Move past the "Ability: Name" line
                        # Collect multi-line ability text
                        # Stop collecting if we hit another Ability, Attack (full pattern), Weakness, Retreat, or ex rule
                        while current_line_index < len(lines):
                            next_line_content = lines[current_line_index]
                            # Use the more specific attack pattern check here
                            is_attack = bool(re.match(r'^([G R W L P F D M Y C N]+)\s+(.*?)\s+(\d+)$', next_line_content))
                            is_ability = next_line_content.startswith("Ability:")
                            is_weakness = next_line_content.startswith("Weakness:")
                            is_retreat = next_line_content.startswith("Retreat:")
                            is_ex_rule = next_line_content.startswith("ex rule:")

                            if is_attack or is_ability or is_weakness or is_retreat or is_ex_rule:
                                break # Stop collecting text
                            ability_text_lines.append(next_line_content)
                            current_line_index += 1
                        ability_text = " ".join(ability_text_lines)
                        card_data['ability'] = {
                            "name": ability_name,
                            "text": ability_text,
                            # Classify as Passive if it starts with "Whenever", otherwise Active
                            "type": "Passive" if ability_text.lower().startswith("whenever") else "Active",
                            "cost": None,
                            "effect_tag": generate_effect_tag("ABILITY", ability_name, ability_text) # Generate tag using collected text
                        }
                        print(f"    Parsed Ability: {ability_name} - Text: {ability_text}")
                        continue # Skip increment at end, already advanced index in inner loop

                    # Check for Attack Start (EnergyCost Name Damage)
                    # Updated regex to handle bracketed costs like [P][P][P][C] or mixed like [G]C
                    attack_match = re.match(r'^((?:\[[G R W L P F D M Y C N]\]|[G R W L P F D M Y C N])+)\s+(.*?)\s+(\d+)$', line)
                    if attack_match:
                        active_attack = None # Finish any previous attack
                        cost_str, attack_name, damage_str = attack_match.groups() # cost_str now contains the raw cost string (e.g., "[P][P][P][C]")
                        damage = int(damage_str)
                        cost = parse_energy_cost(cost_str)
                        active_attack = {
                            "name": attack_name.strip(),
                            "cost": cost,
                            "damage": damage,
                            "effect_text": None, # Will be filled by next line(s)
                            "effect_tag": None
                        }
                        card_data['attacks'].append(active_attack)
                        print(f"    Parsed Attack Start: {attack_name} (Cost: {cost_str}, Dmg: {damage})")

                    # Check for Effect Text (belongs to previous attack or trainer)
                    elif active_attack is not None: # If we just defined an attack start
                        # Assume this line is the effect text
                        active_attack["effect_text"] = line
                        active_attack["effect_tag"] = generate_effect_tag("ATTACK", active_attack["name"], active_attack["effect_text"]) # Generate tag based on collected text
                        print(f"      Attack Effect: {active_attack['effect_text']}")
                        active_attack = None # Effect line consumed, stop associating next lines with this attack
                    elif card_data.get('card_type') in ["Supporter", "Item", "Tool", "Stadium", "Energy"]:
                         # Assume this line starts the Trainer/Energy effect text
                         effect_lines = [line]
                         current_line_index += 1
                         # Collect multi-line effect text
                         while current_line_index < len(lines) and \
                               not lines[current_line_index].startswith("Weakness:") and \
                               not lines[current_line_index].startswith("Retreat:") and \
                               not lines[current_line_index].startswith("ex rule:") and \
                               not re.match(r'^[G R W L P F D M Y C N]+', lines[current_line_index]): # Stop if next line looks like an attack cost
                             effect_lines.append(lines[current_line_index])
                             current_line_index += 1
                         card_data['effect_text'] = " ".join(effect_lines)
                         card_data['effect_tag'] = generate_effect_tag(f"TRAINER_{card_data['card_type'].upper()}", card_data['name'], card_data['effect_text'])
                         print(f"    Parsed Trainer Effect: {card_data['effect_text']}")
                         continue # Skip increment at end of loop as we already advanced index
                    else:
                        print(f"    Warning: Unrecognized line format: {line}")


                    current_line_index += 1

                # --- Weakness / Retreat (Pokemon only) ---
                if card_data.get('card_type') == "Pokemon":
                    while current_line_index < len(lines):
                        line = lines[current_line_index]
                        if line.startswith("Weakness:"):
                            card_data['weakness_type'] = line.split(":", 1)[1].strip()
                            print(f"  Weakness: {card_data['weakness_type']}")
                        elif line.startswith("Retreat:"):
                            try:
                                card_data['retreat_cost'] = int(line.split(":", 1)[1].strip())
                                print(f"  Retreat: {card_data['retreat_cost']}")
                            except ValueError:
                                print(f"  Warning: Could not parse retreat cost: {line}")
                                card_data['retreat_cost'] = None
                        # Ignore 'ex rule' line
                        current_line_index += 1

                # --- Final Checks & Add ---
                if card_data.get('name'): # Basic check if card was parsed
                    # Remove None values for cleaner JSON
                    cleaned_card_data = {k: v for k, v in card_data.items() if v is not None and v != []}
                    all_cards_data.append(cleaned_card_data)
                else:
                    print("  Warning: Failed to parse card name, skipping block.")

            except Exception as e:
                print(f"  ERROR processing block {i+1} for card '{card_data.get('name', 'UNKNOWN')}': {e}")
                traceback.print_exc()
                print(f"  Problematic block content:\n{block}")


        # --- Output to JSON ---
        print(f"\nAttempting to write {len(all_cards_data)} parsed cards to {output_filepath}...")
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(all_cards_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully wrote JSON data to {output_filepath}")
        except Exception as e:
            print(f"ERROR writing JSON file: {e}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        traceback.print_exc()

# --- Main Execution ---
if __name__ == "__main__":
    parse_raw_data()
    print("\nParsing script finished.")
