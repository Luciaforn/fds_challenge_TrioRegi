from helper_feature_extractor import _ALL_TYPES, _get_pokemon_types, _type_multiplier, _norm_type, get_effective_gen1_speed
import numpy  as np
import pandas as pd


def feature_extractor(data, pokedex):
    df_rows = []
    stat_keys = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    boost_keys = ['atk', 'def', 'spa', 'spd', 'spe'] 

    for battle in data: 
        row = {}
        row['battle_id'] = battle.get('battle_id', f"battle_{len(df_rows)}")
        row['player_won'] = battle.get('player_won', None)

        # === 1️⃣ FEATURE STATICHE (P1 Team) ===
        p1_team = battle.get('p1_team_details', [])
        p1_stats = {s: [] for s in stat_keys}
        p1_types = []
        for p in p1_team:
            for s in stat_keys:
                p1_stats[s].append(p.get(s, 0))
            p1_types.extend([t.lower() for t in p.get('types', []) if t and t.lower() in _ALL_TYPES])
        for s, values in p1_stats.items():
            row[f'p1_team_{s}_mean'] = np.mean(values) if values else 0
            row[f'p1_team_{s}_sum'] = np.sum(values) if values else 0
        for t in _ALL_TYPES:
            row[f'p1_type_{t}_count'] = p1_types.count(t)
        row['p1_type_diversity'] = len(set(p1_types))

        # === 2️⃣ FEATURE STATICHE (P2 Lead) ===
        p2_lead = battle.get('p2_lead_details', {})
        p2_lead_stats = {}
        p2_lead_types = []
        p2_lead_name = p2_lead.get('name', 'unknown').lower()
        for s in stat_keys:
            stat_val = p2_lead.get(s, 0)
            row[f'p2_lead_{s}'] = stat_val
            p2_lead_stats[s] = stat_val 
        p2_lead_types = [t.lower() for t in p2_lead.get('types', []) if t and t.lower() in _ALL_TYPES]
        for t in _ALL_TYPES:
            row[f'p2_lead_type_{t}_count'] = p2_lead_types.count(t)
        row['p2_lead_type_diversity'] = len(set(p2_lead_types))

        # === 3️⃣ FEATURE DI BILANCIAMENTO (P1-Team vs P2-Lead) ===
        p1_total_sum = 0
        p2_lead_total_sum = 0
        for s in stat_keys:
            p1_mean = row[f'p1_team_{s}_mean']
            p2_lead_stat = p2_lead_stats.get(s, 0)
            row[f'diff_p1_mean_vs_p2_lead_{s}'] = p1_mean - p2_lead_stat
            p1_total_sum += row[f'p1_team_{s}_sum']
            p2_lead_total_sum += p2_lead_stat
        row['team_vs_lead_power_ratio'] = p1_total_sum / (1 + p2_lead_total_sum) 
        row['type_diversity_diff'] = row['p1_type_diversity'] - row['p2_lead_type_diversity'] 

        # === 4️⃣ FEATURE DINAMICHE (Timeline) ===
        timeline = battle.get('battle_timeline', [])

        p1_hp_list = []
        p2_hp_list = []
        p1_move_cats = {'PHYSICAL': 0, 'SPECIAL': 0, 'STATUS': 0}
        p2_move_cats = {'PHYSICAL': 0, 'SPECIAL': 0, 'STATUS': 0}

        # NEW: momentum/flow trackers
        p1_stab_count = p2_stab_count = 0
        p1_dmg_count = p2_dmg_count = 0
        p1_eff_sum = p2_eff_sum = 0.0
        p1_se = p1_ne = p1_imm = p1_neu = 0
        p2_se = p2_ne = p2_imm = p2_neu = 0

        # Damage estimation on damaging turns (only when same defender stays)
        p1_damage_sum = 0.0
        p2_damage_sum = 0.0

        # KO counts (observed in timeline)
        p1_kos_given = 0
        p1_kos_taken = 0

        # Fresh status inflictions
        p1_inflicted_par = p1_inflicted_slp = p1_inflicted_frz = 0
        p2_inflicted_par = p2_inflicted_slp = p2_inflicted_frz = 0

        # Partial trapping presence
        trapping_moves = {'wrap', 'firespin', 'bind', 'clamp'}
        p1_trap_turns = 0
        p2_trap_turns = 0

        # Early-phase per-turn vectors (first 10 turns) for cheap summaries
        early_turn_cap = 10
        early_p1_eff_list = []
        early_p2_eff_list = []
        early_p1_stab = 0
        early_p2_stab = 0
        early_p1_dmg = 0
        early_p2_dmg = 0

        # For detecting fresh statuses & damage deltas we need previous snapshot
        prev_p1_state = None
        prev_p2_state = None

        # P2 known team discovery
        p2_seen_names = set()
        if p2_lead_name != 'unknown':
            p2_seen_names.add(p2_lead_name)
        turns_seen = {}

        if timeline:
            for turn in timeline[:30]:
                p1_state = turn.get('p1_pokemon_state', {})
                p2_state = turn.get('p2_pokemon_state', {})
                p1_move = turn.get('p1_move_details')
                p2_move = turn.get('p2_move_details')
                p1_active_name = p1_state.get('name', '').lower()
                p2_active_name = p2_state.get('name', '').lower()

                # lists for averages/min
                p1_hp_list.append(p1_state.get('hp_pct', 1.0))
                p2_hp_list.append(p2_state.get('hp_pct', 1.0))

                # move category counts
                if p1_move and p1_move.get('category') in p1_move_cats:
                    p1_move_cats[p1_move.get('category')] += 1
                if p2_move and p2_move.get('category') in p2_move_cats:
                    p2_move_cats[p2_move.get('category')] += 1

                if p1_active_name in pokedex and p2_active_name in pokedex:
                    p1_base_spe = pokedex[p1_active_name]['stats'].get('base_spe', 0)
                    p2_base_spe = pokedex[p2_active_name]['stats'].get('base_spe', 0)

                    p1_boost = p1_state.get('boosts', {}).get('spe', 0)
                    p2_boost = p2_state.get('boosts', {}).get('spe', 0)

                    p1_par = p1_state.get('status') == 'par'
                    p2_par = p2_state.get('status') == 'par'

                    p1_eff_spe = get_effective_gen1_speed(p1_base_spe, p1_boost, p1_par)
                    p2_eff_spe = get_effective_gen1_speed(p2_base_spe, p2_boost, p2_par)

                    if p1_eff_spe > p2_eff_spe:
                        p1_speed_advantage_turns += 1
                    elif p2_eff_spe > p1_eff_spe:
                        p2_speed_advantage_turns += 1

                # --- STAB + effectiveness (damaging only) ---
                # P1 attacking -> defender is P2 active
                if p1_move:
                    move_cat = (p1_move.get('category') or '').upper()
                    move_type = _norm_type(p1_move.get('type', ''))
                    move_name = (p1_move.get('name') or '').lower()
                    if move_cat != 'STATUS':
                        p1_dmg_count += 1
                        atk_types = _get_pokemon_types(p1_state.get('name', ''), pokedex)
                        if move_type in atk_types:
                            p1_stab_count += 1
                            if len(early_p1_eff_list) < early_turn_cap:
                                early_p1_stab += 1
                        def_types = _get_pokemon_types(p2_state.get('name', ''), pokedex)
                        eff = _type_multiplier(move_type, def_types)
                        p1_eff_sum += eff
                        if len(early_p1_eff_list) < early_turn_cap:
                            early_p1_eff_list.append(eff)
                            early_p1_dmg += 1
                        if eff == 0.0: p1_imm += 1
                        elif eff > 1.0: p1_se += 1
                        elif eff < 1.0: p1_ne += 1
                        else: p1_neu += 1

                        # trapping
                        if move_name in trapping_moves:
                            p1_trap_turns += 1

                        # damage estimate (only if same defender species as previous turn)
                        if prev_p2_state and prev_p2_state.get('name') == p2_state.get('name'):
                            prev_hp = prev_p2_state.get('hp_pct', 1.0)
                            cur_hp = p2_state.get('hp_pct', 1.0)
                            delta = max(0.0, prev_hp - cur_hp)
                            p1_damage_sum += delta

                # P2 attacking -> defender is P1 active
                if p2_move:
                    move_cat = (p2_move.get('category') or '').upper()
                    move_type = _norm_type(p2_move.get('type', ''))
                    move_name = (p2_move.get('name') or '').lower()
                    if move_cat != 'STATUS':
                        p2_dmg_count += 1
                        atk_types = _get_pokemon_types(p2_state.get('name', ''), pokedex)
                        if move_type in atk_types:
                            p2_stab_count += 1
                            if len(early_p2_eff_list) < early_turn_cap:
                                early_p2_stab += 1
                        def_types = _get_pokemon_types(p1_state.get('name', ''), pokedex)
                        eff = _type_multiplier(move_type, def_types)
                        p2_eff_sum += eff
                        if len(early_p2_eff_list) < early_turn_cap:
                            early_p2_eff_list.append(eff)
                            early_p2_dmg += 1
                        if eff == 0.0: p2_imm += 1
                        elif eff > 1.0: p2_se += 1
                        elif eff < 1.0: p2_ne += 1
                        else: p2_neu += 1

                        if move_name in trapping_moves:
                            p2_trap_turns += 1

                        if prev_p1_state and prev_p1_state.get('name') == p1_state.get('name'):
                            prev_hp = prev_p1_state.get('hp_pct', 1.0)
                            cur_hp = p1_state.get('hp_pct', 1.0)
                            delta = max(0.0, prev_hp - cur_hp)
                            p2_damage_sum += delta

                # fresh status inflictions (who caused a new non-nostatus on the opponent this turn)
                prev_p1_status = prev_p1_state.get('status', 'nostatus') if prev_p1_state else 'nostatus'
                prev_p2_status = prev_p2_state.get('status', 'nostatus') if prev_p2_state else 'nostatus'
                cur_p1_status = p1_state.get('status', 'nostatus')
                cur_p2_status = p2_state.get('status', 'nostatus')
                if cur_p2_status != 'nostatus' and prev_p2_status == 'nostatus' and p1_move:
                    if cur_p2_status == 'par': p1_inflicted_par += 1
                    elif cur_p2_status == 'slp': p1_inflicted_slp += 1
                    elif cur_p2_status == 'frz': p1_inflicted_frz += 1
                if cur_p1_status != 'nostatus' and prev_p1_status == 'nostatus' and p2_move:
                    if cur_p1_status == 'par': p2_inflicted_par += 1
                    elif cur_p1_status == 'slp': p2_inflicted_slp += 1
                    elif cur_p1_status == 'frz': p2_inflicted_frz += 1

                # KO detection
                # Se un lato è a 0.0 e 'fnt', contiamo chi ha "dato" il KO in questo turno.
                if p1_state.get('hp_pct', 1.0) == 0.0 and p1_state.get('status') == 'fnt' and p2_move:
                    p1_kos_taken += 1
                if p2_state.get('hp_pct', 1.0) == 0.0 and p2_state.get('status') == 'fnt' and p1_move:
                    p1_kos_given += 1

                # P2 discovery
                name = p2_state.get('name')
                if name:
                    lower_name = name.lower()
                    if lower_name not in p2_seen_names:
                        turns_seen[lower_name] = turn['turn']
                    p2_seen_names.add(lower_name)

                # advance prev states
                prev_p1_state = p1_state
                prev_p2_state = p2_state

            # snapshot final
            last_turn = timeline[-1]
            last_p1_state = last_turn.get('p1_pokemon_state', {})
            last_p2_state = last_turn.get('p2_pokemon_state', {})
            last_p1_hp = last_p1_state.get('hp_pct', p1_hp_list[-1] if p1_hp_list else 1.0)
            last_p2_hp = last_p2_state.get('hp_pct', p2_hp_list[-1] if p2_hp_list else 1.0)
            last_p1_status = last_p1_state.get('status', 'nostatus')
            last_p2_status = last_p2_state.get('status', 'nostatus')
            last_p1_boosts = last_p1_state.get('boosts', {b:0 for b in boost_keys})
            last_p2_boosts = last_p2_state.get('boosts', {b:0 for b in boost_keys})
        else:
            last_p1_hp = 1.0
            last_p2_hp = 1.0
            last_p1_status = 'nostatus'
            last_p2_status = 'nostatus'
            last_p1_boosts = {b:0 for b in boost_keys}
            last_p2_boosts = {b:0 for b in boost_keys}
            turns_seen = {}

        # --- Feature dinamiche aggregate (preesistenti + nuove) ---
        row['p1_hp_avg'] = np.mean(p1_hp_list) if p1_hp_list else 1.0
        row['p2_hp_avg'] = np.mean(p2_hp_list) if p2_hp_list else 1.0
        row['p1_hp_min'] = np.min(p1_hp_list) if p1_hp_list else 1.0
        row['p2_hp_min'] = np.min(p2_hp_list) if p2_hp_list else 1.0
        for cat, count in p1_move_cats.items():
            row[f'p1_move_{cat}_count'] = count
        for cat, count in p2_move_cats.items():
            row[f'p2_move_{cat}_count'] = count

        row['p2_avg_turn_first_seen'] = np.mean(list(turns_seen.values())) if turns_seen else 30 
        row['last_p1_hp'] = last_p1_hp
        row['last_p2_hp'] = last_p2_hp
        row['last_hp_diff'] = last_p1_hp - last_p2_hp
        row[f'last_p1_status_{last_p1_status}'] = 1
        row[f'last_p2_status_{last_p2_status}'] = 1
        for b in boost_keys:
            p1_b_val = last_p1_boosts.get(b, 0)
            p2_b_val = last_p2_boosts.get(b, 0)
            row[f'last_p1_boost_{b}'] = p1_b_val
            row[f'last_p2_boost_{b}'] = p2_b_val
            row[f'last_boost_diff_{b}'] = p1_b_val - p2_b_val

        # --- NEW: expose STAB & Effectiveness summary features ---
        row['p1_stab_moves'] = p1_stab_count
        row['p2_stab_moves'] = p2_stab_count
        row['p1_damaging_moves'] = p1_dmg_count
        row['p2_damaging_moves'] = p2_dmg_count
        row['p1_effectiveness_avg'] = (p1_eff_sum / p1_dmg_count) if p1_dmg_count > 0 else 1.0
        row['p2_effectiveness_avg'] = (p2_eff_sum / p2_dmg_count) if p2_dmg_count > 0 else 1.0
        row['eff_avg_diff'] = row['p1_effectiveness_avg'] - row['p2_effectiveness_avg']
        row['p1_super_effective'] = p1_se
        row['p1_not_very_effective'] = p1_ne
        row['p1_immune'] = p1_imm
        row['p1_neutral'] = p1_neu
        row['p2_super_effective'] = p2_se
        row['p2_not_very_effective'] = p2_ne
        row['p2_immune'] = p2_imm
        row['p2_neutral'] = p2_neu

        # --- NEW: momentum & early game ---
        row['p1_damage_sum'] = p1_damage_sum
        row['p2_damage_sum'] = p2_damage_sum
        row['damage_diff'] = p1_damage_sum - p2_damage_sum

        row['p1_kos_given'] = p1_kos_given
        row['p1_kos_taken'] = p1_kos_taken
        row['ko_diff'] = p1_kos_given - p1_kos_taken

        row['p1_inflicted_par'] = p1_inflicted_par
        row['p1_inflicted_slp'] = p1_inflicted_slp
        row['p1_inflicted_frz'] = p1_inflicted_frz
        row['p2_inflicted_par'] = p2_inflicted_par
        row['p2_inflicted_slp'] = p2_inflicted_slp
        row['p2_inflicted_frz'] = p2_inflicted_frz
        row['status_inflict_diff'] = (p1_inflicted_par + p1_inflicted_slp + p1_inflicted_frz) - (p2_inflicted_par + p2_inflicted_slp + p2_inflicted_frz)

        row['p1_trap_turns'] = p1_trap_turns
        row['p2_trap_turns'] = p2_trap_turns
        row['trap_turns_diff'] = p1_trap_turns - p2_trap_turns

        # Early 10 turns aggregates
        row['early10_p1_eff_avg'] = (np.mean(early_p1_eff_list) if early_p1_eff_list else 1.0)
        row['early10_p2_eff_avg'] = (np.mean(early_p2_eff_list) if early_p2_eff_list else 1.0)
        row['early10_eff_diff'] = row['early10_p1_eff_avg'] - row['early10_p2_eff_avg']
        row['early10_p1_stab'] = early_p1_stab
        row['early10_p2_stab'] = early_p2_stab
        row['early10_stab_diff'] = early_p1_stab - early_p2_stab
        row['early10_p1_dmg_turns'] = early_p1_dmg
        row['early10_p2_dmg_turns'] = early_p2_dmg
        row['early10_dmg_turns_diff'] = early_p1_dmg - early_p2_dmg
        row['p1_speed_advantage_turns'] = p1_speed_advantage_turns
        row['p2_speed_advantage_turns'] = p2_speed_advantage_turns
        row['speed_adv_turn_diff'] = p1_speed_advantage_turns - p2_speed_advantage_turns

        # === 5️⃣ P2 squadra rivelata (basata su Pokédex) ===
        p2_known_stats_by_key = {s: [] for s in stat_keys}
        p2_known_types = []
        for name in p2_seen_names:
            if name in pokedex:
                pkmn_data = pokedex[name]
                for s in stat_keys:
                    p2_known_stats_by_key[s].append(pkmn_data['stats'].get(s, 0))
                p2_known_types.extend([t for t in pkmn_data['types'] if t in _ALL_TYPES])
            elif name == p2_lead_name:
                for s in stat_keys:
                    p2_known_stats_by_key[s].append(p2_lead_stats[s])
                p2_known_types.extend(p2_lead_types)

        for s, values in p2_known_stats_by_key.items():
            row[f'p2_known_team_{s}_mean'] = np.mean(values) if values else 0
            row[f'p2_known_team_{s}_sum'] = np.sum(values) if values else 0

        for t in _ALL_TYPES:
            row[f'p2_known_team_type_{t}_count'] = p2_known_types.count(t)
        row['p2_known_team_type_diversity'] = len(set(p2_known_types))
        row['p2_known_team_size'] = len(p2_known_stats_by_key[stat_keys[0]])

        # === 6️⃣ Bilanciamento P1 vs P2-Known ===
        for s in stat_keys:
            p1_mean = row[f'p1_team_{s}_mean']
            p2_known_mean = row[f'p2_known_team_{s}_mean']
            row[f'diff_p1_mean_vs_p2_known_mean_{s}'] = p1_mean - p2_known_mean if p2_known_mean > 0 else p1_mean

        row['known_team_type_diversity_diff'] = row['p1_type_diversity'] - row['p2_known_team_type_diversity'] 
        

        p1_total_stats_sum = sum(row[f'p1_team_{s}_sum'] for s in stat_keys)
        p2_known_stats_sum = sum(row[f'p2_known_team_{s}_sum'] for s in stat_keys)
        row['p1_vs_p2_known_power_ratio'] = p1_total_stats_sum / (1 + p2_known_stats_sum)

        df_rows.append(row)
    with pd.option_context("future.no_silent_downcasting", True):
        return pd.DataFrame(df_rows).infer_objects(copy=False)


