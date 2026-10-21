"""
Paisagem virtual para estimar o número mínimo de amostras de solo necessário
para predizer o estoque de carbono orgânico do solo (ECOS, 0-30 cm) a partir de
índices espectrais de ARP (MicaSense RedEdge-P) com aprendizado de máquina.

Fluxo, para cada área (VN, ILP, IPF):
 1. Lê o ortomosaico (6 bandas: azul, verde, pan, vermelho, red edge, NIR),
    agrega para pixels de ~2,5 m e recorta o domínio amostrado
    (envoltória convexa dos pontos + 15 m).
 2. Calcula os 12 índices espectrais do manuscrito em todos os pixels.
 3. Define a "parte explicável" do ECOS: combinação linear dos índices cuja
    direção é estimada nos pontos observados (regressão ridge), padronizada.
 4. Gera o ECOS verdadeiro de cada pixel:
        z = sqrt(R2) * sinal + sqrt(1 - R2) * resíduo espacial
    em que o resíduo é um campo gaussiano com covariância exponencial e o R2
    controla a força do sinal (cenários: observado, 0,3, 0,5, 0,7).
    Em seguida, z é transformado por quantis para ter a mesma distribuição dos
    estoques observados da área (anamorfose).
 5. Amostragem virtual: sorteia n pixels (aleatório simples), treina Ridge e
    Random Forest e avalia contra o mapa verdadeiro (todos os pixels).
    Repete REPS vezes para cada n e cenário.

    n mínimo = menor n em que pelo menos 80% das repetições atingem 80% do
    R2 máximo possível no cenário (R2_teto).

Saídas: resultados_simulacao.csv, curvas_resumo.csv, resumo_n_minimo.csv,
        parametros_paisagem.csv, learning_curves.png/.pdf
Uso: python paisagem_virtual.py <pasta_rasters> <csv_observados> <pasta_saida>
"""
import sys, os, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.enums import Resampling
from shapely.vectorized import contains
import gstools as gs
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings('ignore')

RAST, OBS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
SEED = 2024
PIX = 2.5                    # tamanho do pixel da paisagem (m)
NS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
CENARIOS = ['observado', 0.3, 0.5, 0.7]
REPS = int(os.environ.get('REPS', 50))
N_AVAL = 5000                # pixels usados para avaliar contra a verdade
FRAC, PROB = 0.8, 0.8         # n mínimo: P(R2 >= FRAC * R2_teto) >= PROB
ALVO = 'solo0-30'
AREAS = {'VN': ('VN_1m.tif', '../points/VN_points.gpkg'),
         'ILP': ('ILP_1m.tif', '../points/ILP_points.gpkg'),
         'IPF': ('IPF_1m.tif', '../points/IPF_points.gpkg')}
IDX = ['EVI', 'SAVI', 'NDVI', 'NDRE', 'PRI', 'sPRI', 'CO2FLUX', 'NDWI', 'SOCI', 'SI', 'CI', 'OI']


def indices(B, G, R, RE, N):
    """Índices espectrais conforme a Tabela 1 do manuscrito."""
    ndvi = (N - R) / (N + R)
    pri = (G - B) / (G + B)
    spri = (pri + 1) / 2
    return {'EVI': 2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1),
            'SAVI': 1.5 * (N - R) / (N + R + 0.5),
            'NDVI': ndvi, 'NDRE': (N - RE) / (N + RE), 'PRI': pri, 'sPRI': spri,
            'CO2FLUX': ndvi * spri, 'NDWI': (G - N) / (G + N),
            'SOCI': B / (R * G), 'SI': (R - B) / (R + B),
            'CI': (R - G) / (R + G), 'OI': (2 * R - G - B) / (R + G)}


def anamorfose(z, alvo):
    """Transforma z (normal padrão) para a distribuição empírica dos observados."""
    q = np.sort(alvo)
    p = (np.arange(1, len(q) + 1) - 0.5) / len(q)
    u = norm.cdf(z)
    # cauda linear além dos extremos observados para não truncar
    return np.interp(u, p, q, left=None, right=None)


def paisagem(area, raster, pontos, obs, rng):
    with rasterio.open(os.path.join(RAST, raster)) as s:
        f = s.res[0] / PIX
        h, w = int(s.height * f), int(s.width * f)
        a = s.read(out_shape=(6, h, w), resampling=Resampling.average, masked=True)
        tr = s.transform * s.transform.scale(s.width / w, s.height / h)
    g = gpd.read_file(os.path.join(RAST, pontos)).to_crs(31983)
    dom = g.unary_union.convex_hull.buffer(15)
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = rasterio.transform.xy(tr, rows.ravel(), cols.ravel())
    xs, ys = np.array(xs), np.array(ys)
    B, G, _, R, RE, N = [a[i].filled(np.nan).ravel().astype(float) for i in range(6)]
    ok = contains(dom, xs, ys) & np.isfinite(B) & (B > 0) & (N > 0)
    I = pd.DataFrame(indices(B[ok], G[ok], R[ok], RE[ok], N[ok]))
    I['x'], I['y'] = xs[ok], ys[ok]
    I = I.replace([np.inf, -np.inf], np.nan).dropna()

    # índices nos pontos observados (pixel da paisagem que contém o ponto)
    from scipy.spatial import cKDTree
    t = cKDTree(I[['x', 'y']].values)
    _, k = t.query(np.c_[g.geometry.x, g.geometry.y])
    Xp = I.iloc[k][IDX].values
    yp = obs[ALVO].values

    # direção do sinal estimada nos observados
    sc = StandardScaler().fit(I[IDX].values)
    rid = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(sc.transform(Xp), yp)
    sinal = sc.transform(I[IDX].values) @ rid.coef_
    sinal = (sinal - sinal.mean()) / sinal.std()
    # R2 "observado": correlação ao quadrado, fora da amostra (LOO), entre sinal e ECOS
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 30)), sc.transform(Xp), yp, cv=LeaveOneOut())
    r2_obs = max(np.corrcoef(pr, yp)[0, 1], 0) ** 2

    # variograma experimental dos resíduos nos pontos (alcance do campo)
    res = yp - rid.predict(sc.transform(Xp))
    pos = np.c_[g.geometry.x, g.geometry.y]
    bins = np.linspace(0, np.ptp(pos, 0).max() * 0.6, 8)
    try:
        bc, gam = gs.vario_estimate((pos[:, 0], pos[:, 1]), res, bins)
        mod = gs.Exponential(dim=2)
        mod.fit_variogram(bc, gam, nugget=True)
        alcance = float(np.clip(mod.len_scale, 10, 300))
        pepita = float(mod.nugget / (mod.nugget + mod.var)) if (mod.nugget + mod.var) > 0 else 0.3
    except Exception:
        alcance, pepita = 50.0, 0.3
    return I, sinal, obs[ALVO].values, r2_obs, alcance, pepita


def main():
    obs_all = pd.read_csv(OBS)
    rng = np.random.default_rng(SEED)
    linhas, info = [], []
    for area, (raster, pontos) in AREAS.items():
        obs = obs_all[obs_all.tratamento == area].reset_index(drop=True)
        I, sinal, yobs, r2o, alc, pep = paisagem(area, raster, pontos, obs, rng)
        info.append(dict(area=area, pixels=len(I), area_ha=len(I) * PIX ** 2 / 1e4,
                         R2_observado=round(r2o, 3), alcance_m=round(alc, 1), pepita_rel=round(pep, 2)))
        print(info[-1], flush=True)
        X = I[IDX].values
        for cen in CENARIOS:
            r2 = r2o if cen == 'observado' else cen
            # resíduo espacial: campo gaussiano exponencial + pepita
            srf = gs.SRF(gs.Exponential(dim=2, var=1, len_scale=alc), seed=int(rng.integers(1e9)))
            campo = srf((I.x.values, I.y.values), mesh_type='unstructured')
            campo = (campo - campo.mean()) / campo.std()
            e = np.sqrt(1 - pep) * campo + np.sqrt(pep) * rng.standard_normal(len(I))
            e = (e - e.mean()) / e.std()
            z = np.sqrt(r2) * sinal + np.sqrt(1 - r2) * e
            y = anamorfose((z - z.mean()) / z.std(), yobs)
            r2_teto = np.corrcoef(sinal, y)[0, 1] ** 2
            aval = rng.choice(len(I), min(N_AVAL, len(I)), replace=False)
            for n in NS:
                for rep in range(REPS):
                    idx = rng.choice(len(I), n, replace=False)
                    for nome, mdl in (('Ridge', make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30)))),
                                      ('RF', RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                                                   random_state=rep, n_jobs=-1))):
                        mdl.fit(X[idx], y[idx])
                        p = mdl.predict(X[aval])
                        linhas.append(dict(area=area, cenario=str(cen), R2_alvo=round(r2, 3),
                                           R2_teto=round(r2_teto, 3), n=n, rep=rep, modelo=nome,
                                           R2=r2_score(y[aval], p),
                                           RMSE=mean_squared_error(y[aval], p) ** 0.5))
            print(area, cen, 'ok', flush=True)
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(OUT, 'resultados_simulacao.csv'), index=False)
    pd.DataFrame(info).to_csv(os.path.join(OUT, 'parametros_paisagem.csv'), index=False)

    # n mínimo: menor n com P(R2 >= FRAC * R2_teto) >= PROB
    g = df.groupby(['area', 'cenario', 'modelo', 'n'])
    s = g.agg(R2_mediana=('R2', 'median'), R2_p10=('R2', lambda v: np.quantile(v, .1)),
              R2_teto=('R2_teto', 'first'), RMSE_mediana=('RMSE', 'median')).reset_index()
    s['P_sucesso'] = g.apply(lambda d: np.mean(d.R2 >= FRAC * d.R2_teto)).values
    s.loc[s.R2_teto < 0.05, 'P_sucesso'] = 0  # sem sinal: nenhum n é suficiente
    s.to_csv(os.path.join(OUT, 'curvas_resumo.csv'), index=False)
    nmin = (s[s.P_sucesso >= PROB].groupby(['area', 'cenario', 'modelo']).n.min()
            .reindex(pd.MultiIndex.from_product([AREAS, [str(c) for c in CENARIOS], ['Ridge', 'RF']],
                                                names=['area', 'cenario', 'modelo'])))
    nmin = nmin.rename('n_minimo').reset_index()
    nmin['n_minimo'] = nmin.n_minimo.map(lambda v: f'>{max(NS)}' if pd.isna(v) else int(v))
    nmin.to_csv(os.path.join(OUT, 'resumo_n_minimo.csv'), index=False)
    print(nmin.to_string())

    plot_curvas(s, df, OUT)


def plot_curvas(s, df, out):
    """Figura das curvas de aprendizado (rótulos em inglês para o artigo)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    nomes = {'VN': 'Native vegetation (NV)', 'ILP': 'Crop-livestock (ICL)', 'IPF': 'Livestock-forest (ILF)'}
    cen = [str(c) for c in CENARIOS]
    cores = dict(zip(cen, ['#7f7f7f', '#9ecae1', '#3182bd', '#08519c']))
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for j, area in enumerate(AREAS):
        for i, mod in enumerate(['Ridge', 'RF']):
            a = ax[i, j]
            for c in cen:
                d = s[(s.area == area) & (s.modelo == mod) & (s.cenario == c)]
                r2o = df[(df.area == area) & (df.cenario == c)].R2_alvo.iloc[0]
                lab = f'Observed (R² = {r2o:.2f})' if c == 'observado' else f'R² = {float(c):.1f}'
                a.plot(d.n, d.R2_mediana, '-o', ms=3, color=cores[c], label=lab)
                a.fill_between(d.n, d.R2_p10, d.R2_mediana, color=cores[c], alpha=.12)
            a.axhline(0, ls='--', lw=.8, color='k')
            a.set_xscale('log'); a.set_xticks(NS); a.set_xticklabels(NS)
            a.set_ylim(-0.6, 0.8); a.grid(alpha=.3)
            a.set_title(f"{nomes[area]} - {'Random Forest' if mod == 'RF' else 'Ridge regression'}", fontsize=10)
            if j == 0: a.set_ylabel('R² against the true map')
            if i == 1: a.set_xlabel('Sample size (n)')
            if i == 0 and j == 2: a.legend(title='Signal strength', fontsize=8)
    fig.suptitle('Learning curves in the virtual landscape (SOC stock, 0-30 cm; median and 10th percentile of 30 repetitions)', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'learning_curves.png'), dpi=300)
    fig.savefig(os.path.join(out, 'learning_curves.pdf'))

if __name__ == '__main__':
    main()
