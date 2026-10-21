"""Observado x predito (validação cruzada) dos modelos Ridge ajustados aos estoques medidos, com R², RMSE e MAPE.
Mesmo esquema da Tabela S3: 5 folds, 20 repetições (random_state = 0..19), mediana das métricas.
A curvatura de perfil não entra (quase constante no modelo de 30 m).
Uso (na raiz do repositório): python scripts/fig_observed_predicted.py data/soc_points.csv results/terrain/atributos_relevo_pontos.csv <pasta_saida>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

OBS, REL, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
IDX = ['EVI', 'SAVI', 'NDVI', 'NDRE', 'PRI', 'sPRI', 'CO2FLUX', 'NDWI', 'SOCI', 'SI', 'CI', 'OI']
TER = ['elevation', 'slope', 'northness', 'eastness', 'plan_curv', 'TPI', 'TWI']
CAMADAS = {'solo0-10': '0–10 cm', 'solo10-20': '10–20 cm', 'solo20-30': '20–30 cm', 'solo0-30': '0–30 cm'}
CONJ = {'Spectral': IDX, 'Terrain': TER, 'Spectral + terrain': IDX + TER, 'Elevation + TWI': ['elevation', 'TWI']}
SITE = {'VN': 'NV', 'ILP': 'ICL', 'IPF': 'ILF'}
COR = {'VN': '#2e7d32', 'ILP': '#c77c0e', 'IPF': '#1f5fa8'}
REPS = 20

obs = pd.read_csv(OBS); rel = pd.read_csv(REL)
obs['ordem'] = obs.groupby('tratamento').cumcount()
DR = obs.merge(rel[['tratamento', 'ordem'] + TER], on=['tratamento', 'ordem'], validate='1:1')

def cv(X, y):
    met, preds = [], []
    for rep in range(REPS):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30)))
        pr = cross_val_predict(m, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=rep))
        met.append((r2_score(y, pr), np.sqrt(mean_squared_error(y, pr)), 100 * mean_absolute_percentage_error(y, pr)))
        preds.append(pr)
    return np.median(met, axis=0), np.mean(preds, axis=0)

tab, PRED = [], {}
for a in SITE:
    g = DR[DR.tratamento == a]
    for c, rot in CAMADAS.items():
        y = g[c].values
        for nome, cols in CONJ.items():
            (r2, rmse, mape), pr = cv(g[cols].values, y)
            tab.append(dict(Site=SITE[a], Layer=rot, Covariates=nome, R2=round(r2, 2), RMSE=round(rmse, 1), MAPE=round(mape, 1),
                            Mean=round(y.mean(), 1)))
            PRED[(a, c, nome)] = (y, pr, r2, rmse, mape)
tab = pd.DataFrame(tab); tab.to_csv(os.path.join(OUT, 'TableS3_cv_metrics.csv'), index=False)

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.linewidth': 0.6})
COLS = ['Spectral', 'Terrain', 'Spectral + terrain']
for c, rot in CAMADAS.items():
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.2), constrained_layout=True)
    for i, a in enumerate(SITE):
        y = PRED[(a, c, COLS[0])][0]
        lo, hi = y.min(), y.max(); pad = 0.12 * (hi - lo); lim = (lo - pad, hi + pad)
        for j, nome in enumerate(COLS):
            y, pr, r2, rmse, mape = PRED[(a, c, nome)]
            A = axs[i, j]
            A.plot(lim, lim, color='0.45', lw=0.8, ls='--', zorder=1)
            A.axhline(y.mean(), color='0.75', lw=0.6, zorder=0)
            A.scatter(y, pr, s=18, color=COR[a], edgecolor='white', linewidth=0.4, zorder=2)
            A.set_xlim(lim); A.set_ylim(lim); A.set_aspect('equal')
            A.text(0.97, 0.04, f'R² = {r2:.2f}\nRMSE = {rmse:.1f} Mg ha⁻¹\nMAPE = {mape:.1f}%', transform=A.transAxes,
                   va='bottom', ha='right', fontsize=6.8, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
            A.text(0.03, 0.97, f'({"abcdefghi"[3 * i + j]})', transform=A.transAxes, ha='left', va='top', fontweight='bold')
            if i == 0: A.set_title(nome, fontsize=8, fontweight='bold')
            if j == 0: A.set_ylabel(f'{SITE[a]}\nPredicted SOC (Mg ha⁻¹)')
            if i == 2: A.set_xlabel('Observed SOC (Mg ha⁻¹)')
            A.spines[['top', 'right']].set_visible(False)
    tag = c.replace('solo', '')
    for e in ('png', 'pdf'): fig.savefig(os.path.join(OUT, f'FigS3_observed_vs_predicted_{tag}.{e}'), dpi=400)
    plt.close(fig)
print(tab[tab.Layer == '0–30 cm'].to_string(index=False))
