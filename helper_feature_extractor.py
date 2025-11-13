import json

_ALL_TYPES = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting',
              'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost',
              'dragon', 'notype']

_TYPE_CHART_GEN1 = {
    'normal':  {'rock': 0.5, 'ghost': 0.0},
    'fire':    {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5},
    'water':   {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
    'electric':{'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0.0, 'flying': 2.0, 'dragon': 0.5},
    'grass':   {'fire': 0.5, 'water': 2.0, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5},
    'ice':     {'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0, 'fire': 0.5},
    'fighting':{'normal': 2.0, 'ice': 2.0, 'rock': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'ghost': 0.0},
    'poison':  {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'bug': 2.0},
    'ground':  {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0.0, 'bug': 0.5, 'rock': 2.0},
    'flying':  {'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'electric': 0.5, 'rock': 0.5},
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'ghost': 0.0},
    'bug':     {'grass': 2.0, 'poison': 2.0, 'psychic': 2.0, 'fire': 0.5, 'fighting': 0.5, 'flying': 0.5, 'ghost': 0.5},
    'rock':    {'fire': 2.0, 'ice': 2.0, 'flying': 2.0, 'bug': 2.0, 'fighting': 0.5, 'ground': 0.5},
    'ghost':   {'normal': 0.0, 'ghost': 2.0, 'psychic': 0.0},
    'dragon':  {'dragon': 2.0}
}

def _norm_type(t):
    if not t:
        return 'notype'
    t = t.lower()
    return t if t in _ALL_TYPES else 'notype'

def _get_pokemon_types(name, pokedex):
    if not name:
        return []
    data = pokedex.get(name.lower(), {})
    types = [tt for tt in data.get('types', []) if tt and tt != 'notype']
    return types

def _type_multiplier(attacking_type, defending_types):
    atk = _norm_type(attacking_type)
    if atk == 'notype':
        return 1.0
    mult = 1.0
    chart_row = _TYPE_CHART_GEN1.get(atk, {})
    for dt in defending_types:
        dt = _norm_type(dt)
        if dt == 'notype':
            continue
        mult *= chart_row.get(dt, 1.0)
    return mult

def build_pokedex(full_data_path):
    """
    Scansiona il dataset completo per creare un dizionario (Pokédex)
    con le statistiche e i tipi di tutti i Pokémon trovati.
    """
    print(f"\nCostruzione Pokédex da '{full_data_path}'...")
    pokedex = {}
    stat_keys = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    try:
        with open(full_data_path, 'r') as f:
            for line in f: 
                try:
                    battle = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pokemon_list = []
                pokemon_list.extend(battle.get('p1_team_details', []))
                p2_lead = battle.get('p2_lead_details', {})
                if p2_lead:
                    pokemon_list.append(p2_lead)
                for p in pokemon_list:
                    name = p.get('name', '').lower()
                    if name and name not in pokedex:
                        stats = {s: p.get(s, 0) for s in stat_keys}
                        types = [t.lower() for t in p.get('types', []) if t]
                        if all(v > 0 for v in stats.values()) and types:
                            pokedex[name] = {'stats': stats, 'types': types}
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare '{full_data_path}' per il Pokédex.")
        exit()
    
    print("Pokedéx scansionato") 
    return pokedex

_GEN1_BOOST_MULTIPLIERS = {
    -6: 0.25, -5: 0.28, -4: 0.33, -3: 0.40, -2: 0.50, -1: 0.66,
     0: 1.0,
     1: 1.5,  2: 2.0,  3: 2.5,  4: 3.0,  5: 3.5,  6: 4.0
}

def get_effective_gen1_speed(base_speed, speed_boost, is_paralyzed):
    if base_speed == 0:
        return 0

    # Applica i boost
    boost_mult = _GEN1_BOOST_MULTIPLIERS.get(max(-6, min(6, speed_boost)), 1.0)
    effective_speed = (base_speed * 2 + 5) * boost_mult # Calcolo statico approssimativo

    # La Paralisi in Gen 1 riduce la velocità del 75% (moltiplica per 0.25)
    if is_paralyzed:
        effective_speed *= 0.25

    return effective_speed