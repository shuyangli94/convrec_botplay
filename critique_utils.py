import random

import numpy as np
import pandas as pd
import torch

# Constants
SMALL_CONST = 1e-15
BIG_CONST = 1e15


def filter_invalid(scores, invalid):
    # Remove invalid items (list of tensor indices)
    batch_size = scores.shape[0]
    n_items = scores.shape[-1]
    n_candidates = [n_items] * batch_size
    if invalid is not None:
        for ix, u_inv in enumerate(invalid):
            scores[ix, u_inv] = -BIG_CONST
            n_candidates[ix] -= len(u_inv)

    return scores, n_candidates


def get_invalid_from_critiques(item_kps: dict, neg: set, pos: set):
    invalid_items = set()
    for i in range(len(item_kps)):
        if i in invalid_items:
            continue
        i_kps = set(item_kps[i].keys())
        # Remove items w negative aspects
        if any(k in i_kps for k in neg):
            invalid_items.add(i)
            continue
        # Remove items w/o positive aspects
        if any(k not in i_kps for k in pos):
            invalid_items.add(i)
            continue
    return invalid_items


def get_gold_rank(scores, item):
    if isinstance(scores, torch.Tensor):
        return torch.argsort(
            torch.argsort(scores, dim=-1, descending=True),
            dim=-1,
            descending=False)[:, item]
    elif isinstance(scores, np.array):
        # item_ranks[b, i] = rank of item i ["rank of item"]
        return np.argsort(np.argsort(-1 * scores, axis=-1), axis=-1)[:, item]
    else:
        raise NotImplementedError('Not supported for input type "{}"'.format(
            type(scores)))


def torch_fillna(tensor, fill_val: float):
    tensor[torch.isnan(tensor)] = fill_val
    return tensor


def compressed_aspects(aspect_bool):
    return [ix for ix, val in enumerate(aspect_bool) if val]


def vector_aspects(aspect_set, n_aspects: int):
    vec = np.zeros(n_aspects, dtype=bool)
    for k in aspect_set:
        vec[k] = True
    return vec


# Target & prediction are a set of aspect indices, NOT boolean vector!
def build_critique(target: set,
                   prediction: set,
                   prev_neg_critiques: set,
                   prev_pos_critiques: set,
                   allow_repeat: bool,
                   allowed_aspects: set,
                   behavior: str,
                   kp_popularity: dict,
                   fb_type: str = 'N',
                   cand_aspect_freq: dict = None,
                   target_aspect_freq: dict = None,
                   allow_arbitrary_critique: bool = False,
                   n_fb: int = 1):
    # Possible negative critiques: in prediction, but not present in gold
    neg_crit_pool = [
        a for a in prediction if a not in target and a in allowed_aspects
    ]
    if not neg_crit_pool and allow_arbitrary_critique:
        neg_crit_pool = [
            a for a in kp_popularity if a not in target and a in allowed_aspects
        ]
    # print('{:,} possible negative critiques ({:,} target, {:,} predicted)'.
    #       format(len(neg_crit_pool), len(target), len(prediction)))

    # Possible positive critiques: predicted and in gold
    pos_crit_pool = [
        a for a in prediction if a in target and a in allowed_aspects
    ]
    if not pos_crit_pool and allow_arbitrary_critique:
        pos_crit_pool = [
            a for a in kp_popularity if a in target and a in allowed_aspects
        ]
    # print('{:,} possible positive critiques ({:,} target, {:,} predicted)'.
    #       format(len(pos_crit_pool), len(target), len(prediction)))

    # Limit repeated critiqued aspects
    if not allow_repeat:
        neg_crit_pool = [
            a for a in neg_crit_pool if a not in prev_neg_critiques
        ]
        pos_crit_pool = [
            a for a in pos_crit_pool if a not in prev_pos_critiques
        ]

    pos_crit = set()
    neg_crit = set()

    try:
        assert behavior in {'random', 'coop', 'uncoop', 'diff'}
    except:
        raise NotImplementedError(
            'Behavior "{}" not supported'.format(behavior))

    # NEGATIVE ONLY
    if fb_type == 'N':
        NCN = min(len(neg_crit_pool), n_fb)
        if NCN:
            if behavior == 'random':
                neg_crit = set(random.sample(neg_crit_pool, NCN))
            # Cooperative - pick the most common negative keyphrases
            elif behavior == 'coop':
                neg_crit = set(
                    sorted(neg_crit_pool,
                           key=lambda x: kp_popularity[x])[-NCN:])
            # Uncooperative - pick the least common negative keyphrases
            elif behavior == 'uncoop':
                neg_crit = set(
                    sorted(neg_crit_pool,
                           key=lambda x: kp_popularity[x])[:NCN])
            # Differential - pick the item w the highest freq difference between target
            # The greater Cand(k) - Tgt(k) is, the more likely to pick
            elif behavior == 'diff':
                neg_crit = set(
                    sorted(
                        neg_crit_pool,
                        key=lambda x: cand_aspect_freq.get(x, 0) - target_aspect_freq.get(x, 0))[-NCN:])

    # POSITIVE ONLY
    elif fb_type == 'P':
        NCP = min(len(pos_crit_pool), n_fb)
        if NCP:
            # Randomly choose
            if behavior == 'random':
                pos_crit = set(random.sample(pos_crit_pool, NCP))
            # Cooperative - pick the rarest positive keyphrases
            elif behavior == 'coop':
                pos_crit = set(
                    sorted(pos_crit_pool,
                           key=lambda x: kp_popularity[x])[:NCP])
            # Uncooperative - pick the most common positive keyphrases
            elif behavior == 'uncoop':
                pos_crit = set(
                    sorted(pos_crit_pool,
                           key=lambda x: kp_popularity[x])[-NCP:])
            # Differential - pick the item w the highest freq difference between target
            # The greater Tgt(k) - Cand(k) is, the more likely to pick
            elif behavior == 'diff':
                pos_crit = set(
                    sorted(
                        pos_crit_pool,
                        key=lambda x: target_aspect_freq.get(x, 0) - cand_aspect_freq.get(x, 0))[-NCP:])
            else:
                raise NotImplementedError(
                    'Behavior "{}" not supported'.format(behavior))

    # MIXED POS/NEGATIVE
    elif fb_type in {'NP', 'PN'}:
        NFB = min(len(pos_crit_pool) + len(neg_crit_pool), n_fb)
        if NFB:
            # Randomly choose
            if behavior == 'random':
                crit_pool = [(pos, 'p') for pos in pos_crit_pool
                             ] + [(neg, 'n') for neg in neg_crit_pool]
                crits = random.sample(crit_pool, NFB)
            # Cooperative - pick the rarest positive keyphrases and most common
            # negative keyphrases.
            # Positive score = positive popularity rank difference over half
            # Negative score = negative popularity rank difference over half
            elif behavior in {'coop', 'uncoop'}:
                # Rank(kp) = Na - 1 if most popular, 0 if least popular
                all_kp_ranks = {
                    kp: ix
                    for ix, (kp, _pop) in enumerate(
                        sorted(
                            [(k, pop) for k, pop in kp_popularity.items()],
                            key=lambda x: x[1]))
                }
                avg_rank = len(all_kp_ranks) / 2.0
                # Positive item score = Rank - 0.5*Na
                # Negative item score = 0.5*Na - Rank
                crit_pool = [(pos, 'p', all_kp_ranks[pos] - avg_rank)
                             for pos in pos_crit_pool
                             ] + [(neg, 'n', avg_rank - all_kp_ranks[neg])
                                  for neg in neg_crit_pool]
                if behavior == 'coop':
                    # Highest score last
                    crits = sorted(crit_pool, key=lambda x: x[-1])[-NFB:]
                elif behavior == 'uncoop':
                    # Lowest score first
                    crits = sorted(crit_pool, key=lambda x: x[-1])[:NFB]
                else:
                    raise NotImplementedError('crits')
                crits = [(kp, typ) for kp, typ, score in crits]

            # Pick aspects with highest absolute difference
            elif behavior == 'diff':
                # Absolute difference score
                crit_pool = [(pos, 'p') for pos in pos_crit_pool
                             ] + [(neg, 'n') for neg in neg_crit_pool]
                # Pickest largest differentials
                crits = sorted(crit_pool, key=lambda x: np.abs(target_aspect_freq.get(x, 0) - cand_aspect_freq.get(x, 0)))[-NFB:]

            pos_crit = {kp for kp, typ in crits if typ == 'p'}
            neg_crit = {kp for kp, typ in crits if typ == 'n'}

    else:
        raise NotImplementedError(
            'Feedback type "{}" not supported'.format(fb_type))

    return pos_crit, neg_crit


def basic_stats(exp_record: dict, verbose: bool = True):
    ##########################################
    # Build aggregate stats from each observation
    agg_stats = []

    window_size = None
    all_durations = []
    for (u, gold), turn_records in exp_record.items():
        turn_df = pd.DataFrame(turn_records)

        if window_size is None:
            window_size = len(turn_df.iloc[0]['recs'])

        # Track original gold rank, # turns w/o critiquing, # turns w/ critiquing
        orig_rank = turn_df.iloc[0]['gold_rank']
        orig_turns = orig_rank // window_size + (1 if orig_rank % window_size
                                                 else 0)
        crit_turns = len(turn_df)
        agg_stats.append({
            'orig_rank': orig_rank,
            'orig_turns': orig_turns,
            'crit_turns': crit_turns,
            'crit_gain': orig_turns - crit_turns
        })
        all_durations.extend(turn_df['time'].values)

    # DataFrame format for easy stats
    adf = pd.DataFrame(agg_stats)

    # Return dict
    overall_stats = dict()

    # Effectiveness in various subsets
    for min_orank, max_orank in [
        (0, 50),
        (50, 200),
        (200, 9999999),
        (0, 9999999),
    ]:
        subdf = adf[(adf['orig_rank'] >= min_orank)
                    & (adf['orig_rank'] < max_orank)]
        if len(subdf) < 1:
            continue
        p_gain = (subdf['crit_gain'] > 0).sum() / len(subdf)
        p_eq = (subdf['crit_gain'] == 0).sum() / len(subdf)
        p_loss = (subdf['crit_gain'] < 0).sum() / len(subdf)
        if verbose:
            print('Rank ({:,} - {:,}): {:.2f}% win, {:.2f}% eq, {:.2f}% loss'.
                  format(min_orank, max_orank, p_gain * 100.0, p_eq * 100.0,
                         p_loss * 100.0))
        # Store overall stats
        if min_orank == 0 and max_orank > 999999:
            overall_stats.update({
                'W': p_gain,
                'E': p_eq,
                'L': p_loss,
            })

    # % of conversations within K turns
    for max_turns in [10, 20, 50]:
        p_orig_at_k = (adf['orig_turns'] < max_turns).sum() / len(adf)
        p_crit_at_k = (adf['crit_turns'] < max_turns).sum() / len(adf)
        if verbose:
            print(
                'Ending <= {:,} turns: {:.2f}% vs. {:.2f}% ({:.2f}% improvement)'.
                format(max_turns, p_orig_at_k * 100.0, p_crit_at_k * 100.0,
                       (p_crit_at_k - p_orig_at_k) * 100.0))
        if max_turns in {10, 20, 50}:
            overall_stats[f'W@{max_turns}'] = p_crit_at_k - p_orig_at_k

    # Aggregate measures
    med_o = adf['orig_turns'].median()
    med_c = adf['crit_turns'].median()
    mean_c = adf['crit_turns'].mean()
    med_g = adf['crit_gain'].median()
    if verbose:
        print(
            'Median {:,.1f} | {:,.1f} turns before/after critiquing ({:,.1f} gain) [Mean {:,.2f}]'.
            format(med_o, med_c, med_g, mean_c))
    overall_stats.update({
        'med_o_turns': med_o,
        'med_c_turns': med_c,
        'mean_c_turns': mean_c,
        'med_gain_turns': med_g
    })

    # Time
    overall_stats['time'] = np.mean(all_durations)
    overall_stats['time_std'] = np.std(all_durations)

    return overall_stats


def build_turn_df(exp_records: dict, window_size: int):
    turn_df = []
    gold_anomalies = 0
    for (u, gold), turn_records in exp_records.items():
        orig_gold = None
        found = False
        prev_gr = None
        for turn_n, t_dict in enumerate(turn_records):
            # Very first gold rank
            if orig_gold is None:
                orig_gold = t_dict['gold_rank']

            # Previous gr
            if prev_gr is None:
                prev_gr = orig_gold

            # Found
            if gold in t_dict['recs']:
                if t_dict['gold_rank'] >= window_size:
                    t_dict['gold_rank'] = window_size - 1
                    gold_anomalies += 1


#                 try:
#                     assert t_dict['gold_rank'] <= window_size
#                 except:
#                     print('Gold: {}'.format(gold))
#                     print('Recs: {}'.format(t_dict['recs']))
#                     print('Gold rank reported: {}'.format(t_dict['gold_rank']))
#                     raise
                found = True

            # What to track
            orig_gr = max(0, orig_gold - window_size * turn_n)
            turn_dict = {
                'turn': turn_n,
                'key': (u, gold),
                'gold': gold,
                'init_GR': orig_gold,
                'orig_GR': orig_gr,
                'crit_GR': t_dict['gold_rank'],
                'RR': 1 / (t_dict['gold_rank'] + 1),
                'max_P': max(t_dict['rec_p']),
                'max_R': max(t_dict['rec_r']),
                'max_F1': max(t_dict['rec_f1']),
                'orig_found': orig_gr < window_size,
                'crit_found': found,
                'delta_GR': prev_gr - t_dict['gold_rank'],
                'delta_GR_p': (prev_gr - t_dict['gold_rank']) / (prev_gr + 1),
            }

            # Derived values
            turn_dict['GR_gain'] = turn_dict['orig_GR'] - turn_dict['crit_GR']
            turn_dict['GR_gain_p'] = turn_dict['GR_gain'] / max(1, orig_gold)

            turn_df.append(turn_dict)
            prev_gr = t_dict['gold_rank']

            if found:
                break

    turn_df = pd.DataFrame(turn_df)
    print('<<{} gold anomalies found while making turn DF>>'.format(
        gold_anomalies))
    return turn_df


def turn_df_stats(turn_df: pd.DataFrame,
                  window_size: int,
                  verbose: bool = True):
    tdf_sorted = turn_df.sort_values(['key', 'turn'])
    hr_rows = []
    rows_ret = []
    stat_ret = dict()

    # Hit rate at each turn
    for found_trace in tdf_sorted.groupby(['key'])['crit_GR'].agg(list):
        for turn_ix, gr in enumerate(found_trace):
            hr_rows.append({'turn': turn_ix, 'gr': gr, 'found': gr < 1})
        # Fill in incomplete rows
        for t in range(turn_ix + 1, 25):
            hr_rows.append({'turn': t, 'gr': gr, 'found': gr < 1})

    # When enforcing turn limits
    for max_turns in [10]:
        if verbose:
            print('== Max {} turn sessions'.format(max_turns))
        limit_df = tdf_sorted[tdf_sorted['turn'] < max_turns].copy()
        last_turn_df = limit_df.groupby(['key']).last().reset_index()

        for at_turns in [1, 5, 10, 20]:
            rank_thresh = at_turns  #* window_size
            # Avg. turns to 1/5/10 within 10/20 turns
            limit_df['under_thresh'] = limit_df['crit_GR'] > rank_thresh
            _tthr = limit_df.groupby(['key'])['under_thresh'].sum()
            # Final target rank <= 1/5/10 within 10/20 turns
            _hr = (last_turn_df['crit_GR'] < rank_thresh).mean()
            if verbose:
                print('{:.2f}% HR@{} (avg. {:.3f} turns, std {:.3f})'.format(
                    _hr * 100.0, at_turns, _tthr.mean(), _tthr.std()))
            # Store
            stat_ret[f'max{max_turns}_HR@{at_turns}'] = _hr * 100.0
            stat_ret[f'max{max_turns}_T@{at_turns}'] = _tthr.mean()
            stat_ret[f'max{max_turns}_sT@{at_turns}'] = _tthr.std()
            rows_ret.append({
                'M': max_turns,
                'N': at_turns,
                'SR': _hr * 100.0,
                'T': _tthr.mean(),
                'sT': _tthr.std(),
            })

    return stat_ret, rows_ret, hr_rows
