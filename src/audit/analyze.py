"""Analyse déterministe des mesures v2 ; aucune sélection de modèle sur ces résultats."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student

from .core import digest, dump, interval, new_output
from .oracle import Oracle


def cluster_summary(values, seed=91000, margin=.001):
    """Une valeur par grappe indépendante ; ne rééchantillonne pas les lignes RL.

    TOST : équivalence à 5 % si l'IC de Student à 90 % est inclus dans ]-margin, margin[.
    La marge est la différence pratique minimale fixée par le protocole avant les mesures.
    """
    a = np.asarray(values, dtype=float)
    result = interval(a)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(len(a), size=(10000, len(a)))].mean(axis=1)
    result['bootstrap_cluster_ci95'] = np.quantile(means, [.025, .975]).tolist()
    result['leave_one_group_out_mean_range'] = [float((a.sum()-a.max())/(len(a)-1)),
                                               float((a.sum()-a.min())/(len(a)-1))]
    se = result['sd']/np.sqrt(len(a))
    half90 = float(student.ppf(.95, len(a)-1)*se)
    result['ci90'] = [result['mean']-half90, result['mean']+half90]
    if se > 0:
        lower = student.sf((result['mean']+margin)/se, len(a)-1)
        upper = student.sf((margin-result['mean'])/se, len(a)-1)
        result['tost_p'] = float(max(lower, upper))
        result['standardized_effect_dz'] = float(result['mean']/result['sd'])
    else:
        result['tost_p'] = float(abs(result['mean']) >= margin)
        result['standardized_effect_dz'] = None
    result['tost_margin'] = margin
    result['equivalent_within_margin_5pct'] = bool(result['tost_p'] < .05)
    return result


def paired(frame, candidate, reference, value='j'):
    p = frame.pivot(index=['group', 'rl_rep'], columns='method', values=value)
    d = (p[candidate]-p[reference]).dropna()
    clustered = d.groupby(level='group').mean()
    return {**cluster_summary(clustered), 'rl_pairs': len(d),
            'unit': 'independent supervised dataset, averaging paired RL replicates',
            'cluster_differences': {str(k): float(v) for k, v in clustered.items()},
            'positive_pairs': int((d > 1e-14).sum()), 'equal_pairs': int((abs(d) <= 1e-14).sum()),
            'negative_pairs': int((d < -1e-14).sum())}


def thresholds(curves, values):
    rows = []
    for run, frame in curves.groupby('run'):
        frame = frame.sort_values('episodes')
        for threshold in values:
            reached = frame[frame.regret <= threshold]
            first = reached.iloc[0] if len(reached) else None
            base = frame.iloc[0]
            rows.append({'run': run, 'group': int(base.group), 'rl_rep': int(base.rl_rep),
                         'n': int(base.n), 'method': base.method, 'threshold': threshold,
                         'success': int(first is not None), 'censoring_horizon': int(frame.episodes.max()),
                         'episodes': int(first.episodes) if first is not None else None,
                         'total_seconds': float(first.total_seconds) if first is not None else None,
                         'total_cards': int(first.total_cards) if first is not None else None,
                         'final_below_threshold': int(frame.iloc[-1].regret <= threshold)})
    return pd.DataFrame(rows)


def matched_costs(curves):
    rows = []
    for resource, budgets in [('total_seconds', [.5, 1., 2., 5., 10., 20.]),
                              ('total_cards', [10000, 50000, 100000, 250000, 500000, 1000000]),
                              ('reused_total_seconds', [.5, 1., 2., 5., 10.])]:
        for run, f in curves.groupby('run'):
            f = f.sort_values('episodes')
            for budget in budgets:
                valid = f[f[resource] <= budget]
                row = valid.iloc[-1] if len(valid) else None
                base = f.iloc[0]
                rows.append({'resource': resource, 'budget': budget, 'run': run,
                             'group': int(base.group), 'rl_rep': int(base.rl_rep), 'method': base.method,
                             'n': int(base.n), 'affordable': row is not None,
                             'episodes': int(row.episodes) if row is not None else None,
                             'spent': float(row[resource]) if row is not None else None,
                             'j': float(row.j) if row is not None else None,
                             'regret': float(row.regret) if row is not None else None,
                             'training_grid_exhausted': bool(row is not None and row.episodes == f.episodes.max())})
    return pd.DataFrame(rows)


def pareto(points):
    costs = points.total_seconds.to_numpy()
    regrets = points.regret.to_numpy()
    keep = [not np.any((costs <= c) & (regrets <= r) & ((costs < c) | (regrets < r)))
            for c, r in zip(costs, regrets, strict=True)]
    return points.loc[keep].sort_values('total_seconds')


def run(folder, output, diagnostics=None, variant=None, replication=None):
    folder = Path(folder)
    completion = json.loads((folder/'completion.json').read_text(encoding='utf-8'))
    if completion['status'] != 'complete':
        raise ValueError('Campagne incomplète.')
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    output = new_output(output, {'phase': 'analysis', 'input': str(folder),
        'runs_sha256': digest(folder/'runs.csv'), 'curves_sha256': digest(folder/'curves.csv'),
        'bootstrap_seed': 91000, 'bootstrap_samples': 10000})
    runs, curves = pd.read_csv(folder/'runs.csv'), pd.read_csv(folder/'curves.csv')
    primary = curves[(curves.episodes == 100000) & (curves.n.isin([0, 20000]))]
    principal = paired(primary, 'logistic', 'standard')
    secondary = curves[curves.group < config['secondary_groups']].copy()
    secondary_at = secondary[(secondary.episodes == 100000) & secondary.n.isin([0, 20000])]
    contrasts = {f'logistic_minus_{method}': paired(secondary_at, 'logistic', method) for method in
        ['frequencies', 'jeffreys', 'exact', 'standard', 'standard_common', 'epsilon_low', 'omega_08', 'noise_005', 'noise_020']}
    surfaces = []
    for n in config['sizes']:
        for episodes in [1000, 5000, 20000, 100000, 200000]:
            f = secondary[(secondary.n.isin([0, n])) & (secondary.episodes == episodes)]
            for method in ['logistic', 'frequencies', 'jeffreys']:
                stats = paired(f, method, 'standard')
                # 'n' de paired compte les grappes ; il ne doit pas écraser la taille supervisée.
                surfaces.append({**stats, 'clusters': stats['n'], 'n': n, 'episodes': episodes, 'method': method})
    dump(output/'surface.json', surfaces)
    first = thresholds(secondary, config['thresholds'])
    first.to_csv(output/'threshold_runs.csv', index=False)
    threshold_summary = []
    for (n, method, threshold), f in first.groupby(['n','method','threshold']):
        successful = f[f.success == 1]
        stats = cluster_summary(f.groupby('group').success.mean())
        threshold_summary.append({'n': int(n), 'method': method, 'threshold': float(threshold),
            'runs': len(f), 'successes': int(f.success.sum()), 'success_rate': stats,
            'median_episodes_conditional_on_success': float(successful.episodes.median()) if len(successful) else None,
            'median_cards_conditional_on_success': float(successful.total_cards.median()) if len(successful) else None,
            'median_seconds_conditional_on_success': float(successful.total_seconds.median()) if len(successful) else None,
            'relapses_at_final_checkpoint': int(((f.success == 1) & (f.final_below_threshold == 0)).sum())})
    costs = matched_costs(secondary[secondary.method.isin(['logistic','frequencies','jeffreys','standard','standard_common'])])
    costs.to_csv(output/'cost_matched_runs.csv', index=False)
    cost_contrasts = []
    for (resource, budget), f in costs.groupby(['resource','budget']):
        for n in config['sizes']:
            cell = f[(f.n == 0) | (f.n == n)]
            for method in ['logistic','frequencies','jeffreys']:
                p = cell.pivot(index=['group','rl_rep'], columns='method', values='j')
                valid = p[[method,'standard']].dropna()
                # Ne présenter de contraste non conditionnel que si toutes les paires sont réalisables.
                record = {'resource': resource, 'budget': float(budget), 'n': n, 'method': method,
                          'affordable_pairs': len(valid), 'required_pairs': 2*config['secondary_groups']}
                if len(valid) == 2*config['secondary_groups']:
                    stats = paired(cell, method, 'standard')
                    record.update({**stats, 'clusters': stats['n'], 'n': n})
                cost_contrasts.append(record)
    dump(output/'cost_contrasts.json', cost_contrasts)
    points = secondary[secondary.method.isin(['logistic','frequencies','jeffreys','standard'])].groupby(
        ['n','method','episodes'], as_index=False)[['total_seconds','total_cards','regret']].mean()
    frontier = pareto(points)
    points.to_csv(output/'quality_cost_points.csv', index=False)
    frontier.to_csv(output/'pareto.csv', index=False)
    finite_errors = []
    oracle = Oracle()
    for r in runs.itertuples():
        with np.load(folder/f'agents/{r.run}.npz', allow_pickle=False) as archive:
            qhit = oracle.vector(archive['Q'][1])
            v = oracle.stand_from(archive['probabilities'])
            qlimit, _ = oracle.optimal(v)
            delta = float(np.max(np.abs(qhit-qlimit)))
            delta_stand = float(np.max(np.abs(oracle.vector(archive['Q'][0])-v)))
            finite_errors.append({'run':r.run, 'group':r.group, 'n':r.n, 'method':r.method,
                'max_hit_error_to_model_limit':delta, 'max_hit_error_to_true_optimum':float(np.max(np.abs(qhit-oracle.q_hit))),
                'max_stand_learning_error': delta_stand,
                'finite_regret_bound': min(2., 2*r.eta+(delta+delta_stand)*r.expected_decisions),
                'regret':r.regret, 'occupancy_identity_error':abs(r.regret-r.occupancy_regret_identity)})
    pd.DataFrame(finite_errors).to_csv(output/'finite_value_errors.csv', index=False)
    # Coût des étapes, séparé des coûts de la campagne entière.
    fit_rows = []
    for path in sorted((folder/'fits').glob('*.json')):
        f = json.loads(path.read_text(encoding='utf-8'))
        fit_rows.append({k:f[k] for k in ['group','n','selected_C','train_generation_seconds','validation_generation_seconds',
            'test_generation_seconds','selected_fit_seconds','selection_seconds','frequency_seconds','smoothing_seconds',
            'direct_stand_seconds','direct_stand_max_error','train_cards','validation_cards','test_cards','metrics_seconds']} |
            {f'stand_build_{m}_seconds': v for m,v in f['stand_build_seconds'].items()})
    fit_frame = pd.DataFrame(fit_rows)
    fit_frame.to_csv(output/'stage_costs.csv', index=False)
    arms = []
    for method, frame in primary.groupby('method'):
        arms.append({'method':method, 'j':cluster_summary(frame.groupby('group').j.mean()),
                     'regret':cluster_summary(frame.groupby('group').regret.mean())})
    summary = {'primary':principal, 'secondary_contrasts':contrasts, 'arms_at_20k_100k':arms,
        'thresholds':threshold_summary, 'config':config, 'completion':completion,
        'cost_contrasts':cost_contrasts, 'max_occupation_identity_error':max(r['occupancy_identity_error'] for r in finite_errors),
        'largest_process_peak_rss_bytes':int(runs.peak_process_rss_bytes.max()),
        'table_bytes':int(runs.table_bytes.max()), 'primary_practical_effect':config['practical_effect'],
        'actual_primary_half_width':(principal['ci95'][1]-principal['ci95'][0])/2}
    if variant:
        vf = pd.read_csv(Path(variant)/'curves.csv')
        vfinal = vf[vf.episodes == 100000]
        summary['variant'] = {'oracle':json.loads((Path(variant)/'oracle_mc.json').read_text(encoding='utf-8')),
            'frequencies_minus_standard':paired(vfinal,'frequencies','standard'),
            'jeffreys_minus_standard':paired(vfinal,'jeffreys','standard'),
            # Ajouté après lecture du contraste à 100 000 épisodes : description, pas test préétabli.
            'descriptive_by_budget':[{'episodes':int(e), 'regret_means':{m:float(f[f.method==m].groupby('group').regret.mean().mean())
                                        for m in ('frequencies','jeffreys','standard')},
                                      **({'frequencies_minus_standard':paired(f,'frequencies','standard')} if e > 0 else {})}
                                     for e, f in vf.groupby('episodes')]}
    if replication:
        summary['replication'] = json.loads((Path(replication)/'comparison.json').read_text(encoding='utf-8'))
    dump(output/'summary.json', summary)
    make_plots(secondary, surfaces, points, frontier, output, diagnostics)
    write_report(summary, fit_frame, output)
    print(json.dumps({'primary':principal, 'secondary':contrasts, 'variant':summary.get('variant'),
                      'seconds':completion['wall_seconds']},indent=2), flush=True)


def make_plots(curves, surfaces, points, frontier, output, diagnostics):
    labels = {'logistic':'Logistique', 'frequencies':'Fréquences', 'jeffreys':'Jeffreys',
              'standard':'Standard (.4)', 'standard_common':'Standard (1)', 'exact':'STAND exacte'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for method, label in labels.items():
        selected = curves[(curves.method == method) & curves.n.isin([0,20000]) & (curves.episodes > 0)]
        groups = selected.groupby(['group','episodes']).regret.mean().reset_index()
        grouped = groups.groupby('episodes').regret
        mean, sd = grouped.mean(), grouped.std()
        axes[0].plot(mean.index, mean.values, label=label)
        axes[0].fill_between(mean.index, np.maximum(mean-sd,1e-7), mean+sd, alpha=.08)
        if method != 'exact':
            p = selected.groupby('episodes')[['total_seconds','total_cards','regret']].mean()
            axes[1].plot(p.total_seconds,p.regret,label=label)
            axes[2].plot(p.total_cards,p.regret,label=label)
    for ax, xlabel in zip(axes,['Épisodes RL','Secondes acquisition + RL','Cartes acquisition + RL'],strict=True):
        ax.set(xscale='log',yscale='log',xlabel=xlabel,ylabel='Regret exact / partie')
        ax.grid(alpha=.2)
    axes[0].set_title('20 000 mains ; moyenne ± 1 écart-type des grappes')
    axes[1].set_title('Coût complet, sélection de C incluse')
    axes[2].set_title('Cartes physiques, pas nombre de parties')
    axes[0].legend(fontsize=8)
    fig.savefig(output/'learning_and_cost.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13,4),layout='constrained')
    data = pd.DataFrame(surfaces)
    # Couleurs saturées à ±0,005 : sinon les écarts décisifs (±0,002) sont invisibles.
    scale = .005
    data['low'] = data.ci95.str[0]
    data['high'] = data.ci95.str[1]
    for ax, method in zip(axes,['logistic','frequencies','jeffreys'],strict=True):
        cell = data[data.method == method]
        p = cell.pivot(index='n',columns='episodes',values='mean')
        lo = cell.pivot(index='n',columns='episodes',values='low')
        hi = cell.pivot(index='n',columns='episodes',values='high')
        im = ax.imshow(p,aspect='auto',cmap='RdBu',vmin=-scale,vmax=scale)
        ax.set(xticks=range(len(p.columns)),xticklabels=[f'{n//1000}k' for n in p.columns],
               yticks=range(len(p.index)),yticklabels=p.index,xlabel='Épisodes RL',ylabel='Mains supervisées',title=labels[method])
        for (i,j),v in np.ndenumerate(p.values):
            mark = '*' if lo.values[i,j] > 0 or hi.values[i,j] < 0 else ''
            ax.text(j,i,f'{v:+.4f}{mark}',ha='center',va='center',fontsize=7,color='white' if abs(v)>.6*scale else 'black')
    fig.colorbar(im,ax=axes,label='J hybride − J standard, couleurs saturées à ±0,005\n* : IC 95 % sur 9 grappes excluant 0')
    fig.savefig(output/'budget_surface.png',dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5),layout='constrained')
    for method in ['logistic','frequencies','jeffreys','standard']:
        p = points[(points.method == method)&(points.total_seconds>0)]
        ax.scatter(p.total_seconds,p.regret,label=labels[method],alpha=.6,s=20)
    f = frontier[frontier.total_seconds>0]
    ax.plot(f.total_seconds,f.regret,'k--',label='Frontière descriptive des points mesurés')
    ax.set(xscale='log',yscale='log',xlabel='Secondes moyennes acquisition + RL',ylabel='Regret moyen exact / partie',
           title='Compromis qualité–coût ; pas une sélection validée de modèle')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(output/'pareto.png',dpi=170); plt.close(fig)
    if diagnostics:
        stats = json.loads((Path(diagnostics)/'kl_summary.json').read_text(encoding='utf-8'))
        fig, axes = plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
        for method,label in [('jeffreys','Jeffreys'),('laplace','Laplace')]:
            rows = [r for r in stats if r['method']==method]
            n = np.array([r['n'] for r in rows])
            mu = np.array([r['finite_realizations_only']['mean'] for r in rows])
            ci = np.array([r['finite_realizations_only']['ci95'] for r in rows])
            axes[0].errorbar(n,n*mu,yerr=np.array([n*(mu-ci[:,0]),n*(ci[:,1]-mu)]),marker='o',label=label,capsize=2)
        axes[0].axhline(25,color='k',ls='--',label='Limite 25 pour le risque lissé')
        axes[0].set(xscale='log',xlabel='Mains n',ylabel='n × KL moyen',title='Risque lissé : IC ponctuels 95 %')
        rows = [r for r in stats if r['method']=='raw']
        axes[1].plot([r['n'] for r in rows],[r['infinite_observations']/2000 for r in rows],marker='o')
        axes[1].set(xscale='log',xlabel='Mains n',ylabel='Fraction des 2 000 KL infinis',
                    title='KL brut : espérance infinie à tout n fini')
        axes[0].legend(fontsize=8)
        for ax in axes: ax.grid(alpha=.2)
        fig.savefig(output/'kl_curve.png',dpi=170); plt.close(fig)


def fmt(stats):
    return f"{stats['mean']:+.7f} [{stats['ci95'][0]:+.7f} ; {stats['ci95'][1]:+.7f}]"


def write_report(s, costs, output):
    p=s['primary']
    lines=['# Mesures v2 — rapport généré depuis les résultats bruts','',
        'Les intervalles entre crochets sont des IC de Student approximatifs à 95 % sur les grappes indépendantes.',
        'Les comparaisons secondaires sont descriptives et leurs intervalles ne sont pas simultanés.','',
        '## Contraste principal préspécifié','',
        f"20 000 mains, 100 000 épisodes : J(logistique) − J(standard TUNED) = **{fmt(p)}**.",
        f"{p['n']} jeux indépendants, {p['rl_pairs']} paires RL. Marge d'intérêt pratique : 0,001 par partie.",
        f"Demi-largeur réalisée : {s['actual_primary_half_width']:.7f}, objectif pilote : 0,0005.",
        f"Bootstrap des grappes : [{p['bootstrap_cluster_ci95'][0]:+.7f} ; {p['bootstrap_cluster_ci95'][1]:+.7f}].",
        f"Médiane des différences de grappes : {p['median']:+.7f} ; étendue : [{p['min']:+.7f} ; {p['max']:+.7f}].",
        f"Moyennes après retrait d'une grappe : {p['leave_one_group_out_mean_range']}.",
        f"Effet standardisé d_z (moyenne / écart-type des grappes) : {p['standardized_effect_dz']:.3f}.", '',
        '| Méthode (20 000 mains si estimée) | J moyen et IC 95 % | Grappes |', '|---|---:|---:|']
    for a in s['arms_at_20k_100k']:
        lines.append(f"| {a['method']} | {fmt(a['j'])} | {a['j']['n']} |")
    lines += ['', '## Contrastes secondaires à 100 000 épisodes', '',
              'TOST : p du test d’équivalence à la marge ±0,001 fixée avant les mesures ; p < 0,05 équivaut',
              'à un IC à 90 % inclus dans la marge. Un p élevé ne prouve ni différence ni équivalence.', '',
              '| Logistique moins référence | Différence et IC 95 % | IC 90 % | p TOST | Paires meilleures / égales / moins bonnes |',
              '|---|---:|---:|---:|---:|']
    for name, stats in s['secondary_contrasts'].items():
        lines.append(f"| {name} | {fmt(stats)} | [{stats['ci90'][0]:+.7f} ; {stats['ci90'][1]:+.7f}] | {stats['tost_p']:.3g} | "
                     f"{stats['positive_pairs']} / {stats['equal_pairs']} / {stats['negative_pairs']} |")
    lines += ['', '![Courbes et coûts](learning_and_cost.png)', '',
              'Les bandes de la première figure sont ±1 écart-type des moyennes de grappes, pas des IC.',
              'Les courbes Fréquences et Jeffreys sont presque confondues ; la première est masquée par la seconde.',
              'Le coût des deux autres panneaux inclut l’acquisition du prédicteur et la sélection de C ;',
              'les évaluations scientifiques sont exclues. Le bras exact est privilégié et omis des coûts.', '',
              '![Surface des budgets](budget_surface.png)', '', '![Frontière descriptive](pareto.png)', '',
              'Les points de la frontière sont des moyennes de configurations préétablies. Les relier ne',
              'prouve pas qu’une configuration interpolée existe, ni que la meilleure a été validée.', '',
              '## Coûts appariés à 20 000 mains', '',
              'Dernier checkpoint dont le coût observé respecte le budget, sans interpolation et sans',
              'choisir rétrospectivement le checkpoint de meilleur J. Une grille arrivée à 200 000 épisodes',
              'n’est pas extrapolée : du budget peut rester inutilisé. La sélection des checkpoints dépend',
              'du temps mesuré ; les secondes restent propres à cette machine et à cette session.', '',
              '| Ressource | Budget | Méthode − standard | Différence et IC 95 % |', '|---|---:|---|---:|']
    for r in s['cost_contrasts']:
        if r['n']==20000 and r['method'] in ['logistic','frequencies'] and 'mean' in r:
            lines.append(f"| {r['resource']} | {r['budget']:g} | {r['method']} | {fmt(r)} |")
    lines += ['', '## Coûts des étapes', '', 'Moyennes entre jeux, tailles 20 000. Pas de somme implicite entre mains et parties.', '',
              '| Poste | Secondes moyennes |', '|---|---:|']
    c = costs[costs.n==20000]
    for name in ['train_generation_seconds','validation_generation_seconds','test_generation_seconds',
                 'selected_fit_seconds','selection_seconds','frequency_seconds','smoothing_seconds','direct_stand_seconds',
                 'stand_build_logistic_seconds','metrics_seconds']:
        lines.append(f"| {name} | {c[name].mean():.6f} |")
    lines += ['', f"Pic RSS du processus : {s['largest_process_peak_rss_bytes']/2**20:.1f} Mio ; tableaux RL instrumentés : {s['table_bytes']} octets.",
              'Le RSS inclut imports, fits et mémoire retenue : il ne mesure pas la mémoire exclusive d’un agent.',
              'Pas de mesure d’énergie. Le coût de recherche RL historique (108 runs) est séparé du coût',
              'd’entraînement d’un agent ; aucune seconde historique absente n’est inventée.', '',
              '## Atteinte du regret 0,002', '',
              '| Méthode | Réussites avant 200k | Médiane épisodes parmi réussites | Médiane cartes parmi réussites |',
              '|---|---:|---:|---:|']
    for r in s['thresholds']:
        if r['threshold']==.002 and r['n'] in [0,20000] and r['method'] in ['standard','logistic','frequencies','jeffreys','exact']:
            lines.append(f"| {r['method']} | {r['successes']}/{r['runs']} | {r['median_episodes_conditional_on_success']} | {r['median_cards_conditional_on_success']} |")
    lines += ['', 'Les échecs sont censurés, pas transformés en réussites à 200k. Les médianes conditionnelles',
              'ne décrivent pas tous les runs. `threshold_runs.csv` conserve les rechutes au dernier point.', '',
              '## Contrôles', '',
              f"Erreur maximale de l’identité de regret pondérée par occupation : {s['max_occupation_identity_error']:.3g}.",
              'Les bornes de politique limite ne sont pas appliquées sans terme d’erreur aux tables finies.',
              'Les erreurs HIT finies et les bornes correspondantes sont dans `finite_value_errors.csv`.', '']
    if 'variant' in s:
        v=s['variant']
        lines += ['## Variante à cartes uniformes', '', f"J optimal de la variante : {v['oracle']['j_star']:+.8f}.",
                  f"Fréquences − standard : {fmt(v['frequencies_minus_standard'])}.",
                  f"Jeffreys − standard : {fmt(v['jeffreys_minus_standard'])}.",
                  f"TOST à ±0,001 (fréquences − standard) : p = {v['frequencies_minus_standard']['tost_p']:.3g}.", '',
                  'Description par budget, ajoutée après lecture du contraste préétabli (IC non simultanés) :', '',
                  '| Épisodes | Regret fréquences | Regret standard | Fréquences − standard |', '|---:|---:|---:|---:|']
        for r in v['descriptive_by_budget']:
            if r['episodes'] > 0:
                lines.append(f"| {r['episodes']} | {r['regret_means']['frequencies']:.5f} | {r['regret_means']['standard']:.5f} | "
                             f"{fmt(r['frequencies_minus_standard'])} |")
        lines += ['',
                  'Les modèles sont réentraînés ; il s’agit d’une robustesse locale, pas d’un transfert sans apprentissage.', '']
    if 'replication' in s:
        lines += ['## Réplication historique', '', 'Résultats des contrôles :', '', '```json',
                  json.dumps(s['replication'],indent=2,ensure_ascii=False), '```', '']
    lines += ['## Données et portée', '',
              '`summary.json`, `surface.json`, `cost_contrasts.json` et les CSV donnent les estimations complètes.',
              'Les graines supervisées sont l’unité indépendante. Les budgets d’un même run et les méthodes',
              'appariées ne sont pas indépendants. La petite taille des analyses secondaires limite la précision.',
              'Un IC contenant zéro ne prouve pas l’équivalence. Aucun modèle livré n’est choisi sur ces résultats.', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--diagnostics',type=Path)
    parser.add_argument('--variant',type=Path)
    parser.add_argument('--replication',type=Path)
    args=parser.parse_args()
    run(args.input,args.output,args.diagnostics,args.variant,args.replication)
