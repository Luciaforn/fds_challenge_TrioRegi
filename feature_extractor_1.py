import pandas as pd
import numpy as np
# Importa tutto dal tuo file helper esistente
from helper_feature_extractor import (
    _ALL_TYPES, 
    _get_pokemon_types, 
    _type_multiplier, 
    _norm_type, 
    get_effective_gen1_speed,
    _GEN1_BOOST_MULTIPLIERS
)

# Costanti per chiarezza
STAT_KEYS = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
BOOST_KEYS = ['atk', 'def', 'spa', 'spd', 'spe']

def _get_initial_stats(pokemon_details):
    """Estrae le statistiche base da un dizionario di dettagli Pokémon."""
    stats = {}
    for s in STAT_KEYS:
        stats[s] = pokemon_details.get(s, 0)
    return stats

def _get_pokedex_stats(pokemon_name, pokedex):
    """Recupera le statistiche dal pokedex (per i Pokémon della timeline)."""
    name = (pokemon_name or '').lower()
    if name in pokedex:
        # Assumiamo che 'stats' in pokedex usi le stesse chiavi di STAT_KEYS
        return pokedex[name].get('stats', {s: 0 for s in STAT_KEYS})
    return {s: 0 for s in STAT_KEYS}

# --- FASE 1: FEATURE STATICHE (LEAD vs LEAD) ---

def get_static_lead_features(p1_lead, p2_lead, pokedex):
    """
    Crea feature basate sul matchup 1v1 iniziale.
    """
    features = {}
    if not p1_lead or not p2_lead:
        return features

    p1_stats = _get_initial_stats(p1_lead)
    p2_stats = _get_initial_stats(p2_lead)
    
    p1_name = p1_lead.get('name')
    p2_name = p2_lead.get('name')

    # Calcola la velocità iniziale (senza boost/paralisi)
    p1_init_speed = get_effective_gen1_speed(p1_stats['base_spe'], 0, False)
    p2_init_speed = get_effective_gen1_speed(p2_stats['base_spe'], 0, False)
    
    features['lead_speed_adv'] = 1 if p1_init_speed > p2_init_speed else (-1 if p2_init_speed > p1_init_speed else 0)
    features['lead_hp_diff'] = p1_stats['base_hp'] - p2_stats['base_hp']
    features['lead_atk_vs_def_diff'] = p1_stats['base_atk'] - p2_stats['base_def']
    features['lead_spa_vs_spd_diff'] = p1_stats['base_spa'] - p2_stats['base_spd']
    features['lead_total_stats_diff'] = sum(p1_stats.values()) - sum(p2_stats.values())

    # Vantaggio di tipo (basato su STAB)
    p1_types = _get_pokemon_types(p1_name, pokedex)
    p2_types = _get_pokemon_types(p2_name, pokedex)
    
    p1_stab_mults = [_type_multiplier(t, p2_types) for t in p1_types if t != 'notype']
    features['p1_lead_max_stab_eff'] = max(p1_stab_mults) if p1_stab_mults else 1.0

    p2_stab_mults = [_type_multiplier(t, p1_types) for t in p2_types if t != 'notype']
    features['p2_lead_max_stab_eff'] = max(p2_stab_mults) if p2_stab_mults else 1.0
    
    features['lead_stab_advantage'] = features['p1_lead_max_stab_eff'] - features['p2_lead_max_stab_eff']
    
    return features

# --- FASE 2: FEATURE STATICHE (TEAM P1 vs LEAD P2) ---

def get_static_team_features(p1_team, p2_lead, pokedex):
    """
    Crea feature basate sulla composizione dell'intero team di P1
    in relazione al lead di P2.
    """
    features = {}
    if not p1_team or not p2_lead:
        return features
        
    all_stats = {s: [] for s in STAT_KEYS}
    all_types = []
    counters_to_p2_lead = 0
    resists_p2_lead_stab = 0
    weak_to_p2_lead_stab = 0

    p2_lead_types = _get_pokemon_types(p2_lead.get('name'), pokedex)
    
    for p in p1_team:
        p_stats = _get_initial_stats(p)
        for s in STAT_KEYS:
            all_stats[s].append(p_stats[s])
            
        p_types = _get_pokemon_types(p.get('name'), pokedex)
        all_types.extend(p_types)
        
        # P1 ha un counter per il lead P2? (STAB superefficace)
        p_stab_mults = [_type_multiplier(t, p2_lead_types) for t in p_types if t != 'notype']
        if any(m > 1.0 for m in p_stab_mults):
            counters_to_p2_lead += 1
            
        # P1 resiste allo STAB del lead P2?
        p_resist_mults = [_type_multiplier(t, p_types) for t in p2_lead_types if t != 'notype']
        if p_resist_mults and all(m < 1.0 for m in p_resist_mults):
            resists_p2_lead_stab += 1
        if any(m > 1.0 for m in p_resist_mults):
            weak_to_p2_lead_stab += 1

    # Statistiche aggregate P1
    for s in STAT_KEYS:
        features[f'p1_team_avg_{s}'] = np.mean(all_stats[s]) if all_stats[s] else 0
        features[f'p1_team_max_{s}'] = np.max(all_stats[s]) if all_stats[s] else 0
        features[f'p1_team_sum_{s}'] = np.sum(all_stats[s]) if all_stats[s] else 0
        
    features['p1_team_avg_total_stats'] = sum(features[f'p1_team_avg_{s}'] for s in STAT_KEYS)
    
    # Copertura P1 vs P2 Lead
    features['p1_team_counters_to_p2_lead'] = counters_to_p2_lead
    features['p1_team_resists_p2_lead_stab'] = resists_p2_lead_stab
    features['p1_team_weak_to_p2_lead_stab'] = weak_to_p2_lead_stab
    
    # Diversità tipi P1
    for t in _ALL_TYPES:
        features[f'p1_team_type_{t}_count'] = all_types.count(_norm_type(t))
    features['p1_team_type_diversity'] = len(set(t for t in all_types if t != 'notype'))

    return features

# --- FASE 3: FEATURE DINAMICHE (TIMELINE) ---

def get_dynamic_timeline_features(timeline, pokedex, p2_lead):
    """
    Estrae feature analizzando la timeline (primi 30 turni).
    Include anche l'identificazione del team P2.
    """
    features = {}
    
    # Team P2 "visto" (inizia con il lead)
    p2_seen_names = set()
    if p2_lead and p2_lead.get('name'):
        p2_seen_names.add(p2_lead.get('name').lower())

    if not timeline:
        # Se non c'è timeline, restituisce solo info base sul team P2 visto (solo il lead)
        p2_known_stats = {s: [] for s in STAT_KEYS}
        p2_known_types = []
        for name in p2_seen_names:
             if name in pokedex:
                p_data = pokedex[name]
                stats = p_data.get('stats', {})
                types = p_data.get('types', [])
                for s in STAT_KEYS:
                    p2_known_stats[s].append(stats.get(s, 0))
                p2_known_types.extend(types)
        
        for s in STAT_KEYS:
            features[f'p2_known_team_avg_{s}'] = np.mean(p2_known_stats[s]) if p2_known_stats[s] else 0
        features['p2_known_team_size'] = len(p2_seen_names)
        
        return features

    # Trackers per il loop
    p1_hp_list = [1.0]
    p2_hp_list = [1.0]
    p1_fainted = 0
    p2_fainted = 0
    p1_status_inflicted = {s: 0 for s in ['par', 'slp', 'frz', 'psn', 'brn']}
    p2_status_inflicted = {s: 0 for s in ['par', 'slp', 'frz', 'psn', 'brn']}
    p1_damage_dealt = 0.0
    p2_damage_dealt = 0.0
    p1_switches = 0
    p2_switches = 0
    p1_crit_count = 0
    p2_crit_count = 0
    p1_speed_adv_turns = 0
    
    prev_p1_state = None
    prev_p2_state = None

    for turn in timeline[:30]: # Limita ai primi 30 turni
        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')

        # Aggiungi P2 al team "visto"
        p2_active_name = p2_state.get('name', '').lower()
        if p2_active_name:
            p2_seen_names.add(p2_active_name)

        # --- Tracking HP ---
        p1_hp_list.append(p1_state.get('hp_pct', 1.0))
        p2_hp_list.append(p2_state.get('hp_pct', 1.0))

        # --- Tracking Switch ---
        if prev_p1_state and p1_state.get('name') != prev_p1_state.get('name'):
            p1_switches += 1
        if prev_p2_state and p2_state.get('name') != prev_p2_state.get('name'):
            p2_switches += 1
            
        # --- Tracking Danno e Crit ---
        # P1 attacca P2
        if p1_move and prev_p2_state and p2_state.get('name') == prev_p2_state.get('name'):
            dmg = max(0.0, prev_p2_state.get('hp_pct', 1.0) - p2_state.get('hp_pct', 1.0))
            p1_damage_dealt += dmg
            if p1_move.get('crit'):
                p1_crit_count += 1
        # P2 attacca P1
        if p2_move and prev_p1_state and p1_state.get('name') == prev_p1_state.get('name'):
            dmg = max(0.0, prev_p1_state.get('hp_pct', 1.0) - p1_state.get('hp_pct', 1.0))
            p2_damage_dealt += dmg
            if p2_move.get('crit'):
                p2_crit_count += 1

        # --- Tracking Status (nuove inflizioni) ---
        cur_p1_status = p1_state.get('status', 'nostatus')
        cur_p2_status = p2_state.get('status', 'nostatus')
        prev_p1_status = prev_p1_state.get('status', 'nostatus') if prev_p1_state else 'nostatus'
        prev_p2_status = prev_p2_state.get('status', 'nostatus') if prev_p2_state else 'nostatus'

        if cur_p2_status != 'nostatus' and cur_p2_status != prev_p2_status and p1_move:
            if cur_p2_status in p1_status_inflicted:
                p1_status_inflicted[cur_p2_status] += 1
        if cur_p1_status != 'nostatus' and cur_p1_status != prev_p1_status and p2_move:
            if cur_p1_status in p2_status_inflicted:
                p2_status_inflicted[cur_p1_status] += 1
                
        # --- Tracking Faint ---
        if cur_p1_status == 'fnt' and prev_p1_status != 'fnt':
            p1_fainted += 1
        if cur_p2_status == 'fnt' and prev_p2_status != 'fnt':
            p2_fainted += 1

        # --- Tracking Vantaggio Velocità (per turno) ---
        p1_stats = _get_pokedex_stats(p1_state.get('name'), pokedex)
        p2_stats = _get_pokedex_stats(p2_state.get('name'), pokedex)
        
        p1_eff_speed = get_effective_gen1_speed(
            p1_stats['base_spe'], 
            p1_state.get('boosts', {}).get('spe', 0), 
            cur_p1_status == 'par'
        )
        p2_eff_speed = get_effective_gen1_speed(
            p2_stats['base_spe'], 
            p2_state.get('boosts', {}).get('spe', 0), 
            cur_p2_status == 'par'
        )
        
        if p1_eff_speed > p2_eff_speed:
            p1_speed_adv_turns += 1

        # Aggiorna stati precedenti
        prev_p1_state = p1_state
        prev_p2_state = p2_state

    # --- Aggregazione Feature Dinamiche ---
    final_p1_state = timeline[-1].get('p1_pokemon_state', {})
    final_p2_state = timeline[-1].get('p2_pokemon_state', {})

    features['final_hp_p1'] = final_p1_state.get('hp_pct', 1.0)
    features['final_hp_p2'] = final_p2_state.get('hp_pct', 1.0)
    features['final_hp_diff'] = features['final_hp_p1'] - features['final_hp_p2']
    
    features['p1_fainted_count'] = p1_fainted
    features['p2_fainted_count'] = p2_fainted
    features['fainted_diff'] = p2_fainted - p1_fainted # Positivo è buono per P1

    features['p1_total_damage_dealt'] = p1_damage_dealt
    features['p2_total_damage_dealt'] = p2_damage_dealt
    features['damage_dealt_ratio'] = p1_damage_dealt / (p2_damage_dealt + 1e-6)

    features['p1_switch_count'] = p1_switches
    features['p2_switch_count'] = p2_switches
    features['p1_crit_count'] = p1_crit_count
    features['p2_crit_count'] = p2_crit_count
    
    features['p1_speed_adv_turns_pct'] = p1_speed_adv_turns / len(timeline)
    
    # Status gravi inflitti
    features['p1_inflicted_par'] = p1_status_inflicted['par']
    features['p1_inflicted_slp_frz'] = p1_status_inflicted['slp'] + p1_status_inflicted['frz']
    features['p2_inflicted_par'] = p2_status_inflicted['par']
    features['p2_inflicted_slp_frz'] = p2_status_inflicted['slp'] + p2_status_inflicted['frz']
    
    features['p1_avg_hp'] = np.mean(p1_hp_list)
    features['p2_avg_hp'] = np.mean(p2_hp_list)

    # Boost finali
    p1_final_boosts = final_p1_state.get('boosts', {})
    p2_final_boosts = final_p2_state.get('boosts', {})
    for b in BOOST_KEYS:
        p1_b = p1_final_boosts.get(b, 0)
        p2_b = p2_final_boosts.get(b, 0)
        features[f'final_boost_diff_{b}'] = p1_b - p2_b
        
    # --- Feature Team P2 "Visto" ---
    p2_known_stats = {s: [] for s in STAT_KEYS}
    p2_known_types = []
    for name in p2_seen_names:
        if name in pokedex:
            p_data = pokedex[name]
            stats = p_data.get('stats', {})
            types = p_data.get('types', [])
            for s in STAT_KEYS:
                p2_known_stats[s].append(stats.get(s, 0))
            p2_known_types.extend(types)

    for s in STAT_KEYS:
        features[f'p2_known_team_avg_{s}'] = np.mean(p2_known_stats[s]) if p2_known_stats[s] else 0
        
    for t in _ALL_TYPES:
        features[f'p2_known_team_type_{t}_count'] = p2_known_types.count(_norm_type(t))
    features['p2_known_team_type_diversity'] = len(set(t for t in p2_known_types if t != 'notype'))
    features['p2_known_team_size'] = len(p2_seen_names)

    return features

# --- FUNZIONE PRINCIPALE (ORCHESTRATORE) ---

def feature_extractor_1(data, pokedex):
    """
    Funzione principale che orchestra l'estrazione delle feature.
    Itera su ogni battaglia ed esegue le 3 fasi di estrazione.
    """
    all_features = []
    
    for i, battle in enumerate(data):
        battle_features = {}
        battle_features['battle_id'] = battle.get('battle_id', f"battle_{i}")
        
        # Aggiungi il target se presente (solo per il train)
        if 'player_won' in battle:
            battle_features['player_won'] = int(battle['player_won'])
            
        # Estrai i dati grezzi
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})
        timeline = battle.get('battle_timeline', [])
        
        # Salta la battaglia se mancano dati fondamentali
        p1_lead = p1_team[0] if p1_team else {}
        if not p1_lead or not p2_lead:
            continue
            
        # --- Esegui Fasi ---
        features_static_lead = get_static_lead_features(p1_lead, p2_lead, pokedex)
        features_static_team = get_static_team_features(p1_team, p2_lead, pokedex)
        features_dynamic = get_dynamic_timeline_features(timeline, pokedex, p2_lead)
        
        battle_features.update(features_static_lead)
        battle_features.update(features_static_team)
        battle_features.update(features_dynamic)
        
        # --- FASE 4: Feature di Confronto (P1-Team vs P2-Known-Team) ---
        # Queste feature usano l'output delle fasi 2 e 3
        
        p1_total_sum = sum(battle_features.get(f'p1_team_sum_{s}', 0) for s in STAT_KEYS)
        p2_known_sum = sum(features_dynamic.get(f'p2_known_team_avg_{s}', 0) for s in STAT_KEYS) * features_dynamic.get('p2_known_team_size', 1)
        
        features['p1_vs_p2_known_power_ratio'] = p1_total_sum / (1 + p2_known_sum)

        for s in STAT_KEYS:
            p1_mean = battle_features.get(f'p1_team_avg_{s}', 0)
            p2_known_mean = battle_features.get(f'p2_known_team_avg_{s}', 0)
            # Calcola la differenza solo se P2 ha statistiche note (p2_known_mean > 0)
            battle_features[f'diff_p1_avg_vs_p2_known_avg_{s}'] = p1_mean - p2_known_mean if p2_known_mean > 0 else p1_mean

        battle_features['known_team_type_diversity_diff'] = battle_features.get('p1_team_type_diversity', 0) - battle_features.get('p2_known_team_type_diversity', 0)
            
        all_features.append(battle_features)
        
    # Crea il DataFrame finale
    df = pd.DataFrame(all_features)
    
    # Pulisci eventuali NaN o Inf derivanti da divisioni per zero
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Riempi i NaN. Per la maggior parte delle feature, 0 è un default sicuro.
    # Potresti voler essere più specifico (es. riempire l'hp medio con 1.0)
    # ma iniziamo con 0.
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.fillna(0).infer_objects(copy=False)
    
    return df