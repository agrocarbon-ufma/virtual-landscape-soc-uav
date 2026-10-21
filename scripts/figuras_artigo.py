"""Versões preliminares das figuras do artigo (paisagem virtual). Uso:
python figuras_artigo.py <pasta_rasters_1m> <csv_observados> <pasta_resultados> <pasta_saida>"""
import sys, os, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.enums import Resampling
from shapely import contains_xy
from scipy.spatial import cKDTree
from scipy.stats import norm
import gstools as gs
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')
RAST, OBS, RES, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6,
                     'axes.edgecolor': '#555', 'xtick.major.width': .6, 'ytick.major.width': .6,
                     'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 400})
INK, MUTED = '#222222', '#6b6b6b'
SOC = LinearSegmentedColormap.from_list('soc', ['#f6efe3', '#d9b98a', '#a8743f', '#6b4423', '#3b2412'])
BLUE = LinearSegmentedColormap.from_list('b', ['#f3f7fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('d', ['#1c5cab', '#86b6ef', '#e9e9e7', '#f2a07c', '#c04a1c'])
AREA_COL = {'VN': '#2a78d6', 'ILP': '#eb6834', 'IPF': '#1baf7a'}
AREA_NAME = {'VN': 'Native vegetation', 'ILP': 'Crop-livestock', 'IPF': 'Livestock-forest'}
IDX = ['EVI', 'SAVI', 'NDVI', 'NDRE', 'PRI', 'sPRI', 'CO2FLUX', 'NDWI', 'SOCI', 'SI', 'CI', 'OI']
PIX = 2.5; rng = np.random.default_rng(7)

def letter(ax, l, x=-0.02, y=1.04):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right', color=INK)

def anam(z, a):
    q = np.sort(a); p = (np.arange(1, len(q) + 1) - .5) / len(q); return np.interp(norm.cdf(z), p, q)

# ---------- paisagem do IPF em grade (para mapas) ----------
with rasterio.open(f'{RAST}/IPF_1m.tif') as s:
    f = s.res[0] / PIX; h, w = int(s.height * f), int(s.width * f)
    a = s.read(out_shape=(6, h, w), resampling=Resampling.average, masked=True).filled(np.nan).astype(float)
    tr = s.transform * s.transform.scale(s.width / w, s.height / h)
g = gpd.read_file(f'{RAST}/../points/IPF_points.gpkg').to_crs(31983)
dom = g.union_all().convex_hull.buffer(15)
cols, rows = np.meshgrid(np.arange(w), np.arange(h))
xs, ys = rasterio.transform.xy(tr, rows.ravel(), cols.ravel()); xs, ys = np.asarray(xs), np.asarray(ys)
B, G, _, R, RE, N = a
inside = contains_xy(dom, xs, ys).reshape(h, w) & np.isfinite(B) & (B > 0)
r0, r1 = np.where(inside.any(1))[0][[0, -1]]; c0, c1 = np.where(inside.any(0))[0][[0, -1]]
sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
m = inside[sl]; X0, Y0 = xs.reshape(h, w)[sl], ys.reshape(h, w)[sl]
b, gg, r, re, n = (v[sl] for v in (B, G, R, RE, N))
ndvi = (n - r) / (n + r); pri = (gg - b) / (gg + b); spri = (pri + 1) / 2
ind = {'EVI': 2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1), 'SAVI': 1.5 * (n - r) / (n + r + .5), 'NDVI': ndvi,
       'NDRE': (n - re) / (n + re), 'PRI': pri, 'sPRI': spri, 'CO2FLUX': ndvi * spri, 'NDWI': (gg - n) / (gg + n),
       'SOCI': b / (r * gg), 'SI': (r - b) / (r + b), 'CI': (r - gg) / (r + gg), 'OI': (2 * r - gg - b) / (r + gg)}
F = np.column_stack([ind[k][m] for k in IDX]); ok = np.isfinite(F).all(1)
obs = pd.read_csv(OBS); obs = obs[obs.tratamento == 'IPF'].reset_index(drop=True)
px, py = X0[m][ok], Y0[m][ok]
_, k = cKDTree(np.c_[px, py]).query(np.c_[g.geometry.x, g.geometry.y])
sc = StandardScaler().fit(F[ok]); rid = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(sc.transform(F[ok][k]), obs['solo0-30'])
sig = sc.transform(F[ok]) @ rid.coef_
if sig.std() == 0:  # sinal nulo nos observados: usa a 1a componente principal só para ilustrar
    from sklearn.decomposition import PCA; sig = PCA(1).fit_transform(sc.transform(F[ok]))[:, 0]
sig = (sig - sig.mean()) / sig.std()
srf = gs.SRF(gs.Exponential(dim=2, var=1, len_scale=60), seed=11)
e = srf((px, py), mesh_type='unstructured'); e = (e - e.mean()) / e.std()
def field(vals):
    out = np.full(m.shape, np.nan); tmp = np.full(m.sum(), np.nan); tmp[ok] = vals; out[m] = tmp; return out
yobs = obs['solo0-30'].values
soc = {r2: anam(np.sqrt(r2) * sig + np.sqrt(1 - r2) * e, yobs) for r2 in (0.0, 0.3, 0.7)}
ext = [X0[0, 0] - PIX / 2, X0[0, -1] + PIX / 2, Y0[-1, 0] - PIX / 2, Y0[0, 0] + PIX / 2]
rgb = np.dstack([r, gg, b]); rgb = np.clip(rgb / np.nanpercentile(rgb[m], 98), 0, 1); rgb[~m] = 1

# ---------- Figura 2: anatomia da paisagem virtual ----------
fig, ax = plt.subplots(3, 2, figsize=(7.2, 5.0), constrained_layout=True)
ax = ax.ravel()
vmin, vmax = np.percentile(yobs, [2, 98])
panels = [('a', 'UAV mosaic (true colour) and field points', None),
          ('b', 'Spectral signal from the 12 indices', field(sig)),
          ('c', 'Spatial residual (Gaussian random field)', field(e)),
          ('d', 'Virtual SOC stock, no spectral signal', field(soc[0.0])),
          ('e', 'Virtual SOC stock, signal R² = 0.7', field(soc[0.7])),
          ('f', 'One virtual survey, n = 50 random points', field(soc[0.7]))]
for i, (l, t, z) in enumerate(panels):
    A = ax[i]
    if z is None:
        A.imshow(rgb, extent=ext); A.scatter(g.geometry.x, g.geometry.y, s=10, c='#ffffff', edgecolors=INK, linewidths=.6, zorder=3)
        A.plot([ext[1] - 130, ext[1] - 30], [ext[2] + 25] * 2, color=INK, lw=1.5)
        A.text(ext[1] - 80, ext[2] + 35, '100 m', ha='center', fontsize=6)
    else:
        sd = l in 'bc'
        im = A.imshow(z, extent=ext, cmap=DIV if sd else SOC, vmin=-2.5 if sd else vmin, vmax=2.5 if sd else vmax)
        cax = A.inset_axes([0.58, 0.10, 0.38, 0.045])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5, length=2)
        cb.set_label('z-score' if sd else 'SOC stock 0–30 cm (Mg ha⁻¹)', fontsize=5.5, labelpad=1)
        if l == 'f':
            idx = rng.choice(len(px), 50, replace=False)
            A.scatter(px[idx], py[idx], s=8, c='#ffffff', edgecolors=INK, linewidths=.5, zorder=3)
    A.set_title(t, loc='left', color=INK); letter(A, l)
    A.set_xticks([]); A.set_yticks([]); [sp.set_visible(False) for sp in A.spines.values()]
fig.savefig(f'{OUT}/Fig2_virtual_landscape_anatomy.png'); fig.savefig(f'{OUT}/Fig2_virtual_landscape_anatomy.pdf'); plt.close(fig)

# ---------- Figura 3: bloco 3D com as três camadas ----------
z = field(soc[0.7]); zn = (z - np.nanmean(z)) / np.nanstd(z)
lay = []
for kk, frac in enumerate([0.42, 0.33, 0.25]):  # proporção aproximada do estoque por camada
    srf2 = gs.SRF(gs.Exponential(dim=2, var=1, len_scale=60), seed=30 + kk)
    e2 = field(srf2((px, py), mesh_type='unstructured')); e2 = (e2 - np.nanmean(e2)) / np.nanstd(e2)
    lay.append(frac * np.nanmean(z) * (1 + 0.18 * (0.8 * zn + 0.6 * e2)))
st = 4
Xs, Ys = (X0 - X0.min())[::st, ::st], (Y0 - Y0.min())[::st, ::st]
relief = 1.5 * np.sin(Xs / 140) + 1.0 * np.cos(Ys / 90)  # relevo ilustrativo (substituir pelo MDT)
fig = plt.figure(figsize=(7.2, 4.2)); A = fig.add_subplot(projection='3d')
allv = np.concatenate([np.ravel(L[::st, ::st]) for L in lay]); lv0, lv1 = np.nanpercentile(allv, [2, 98])
for kk, (L, name) in enumerate(zip(lay, ['0–10 cm', '10–20 cm', '20–30 cm'])):
    Z = relief - kk * 14; C = L[::st, ::st]
    fc = SOC((C - lv0) / (lv1 - lv0)); fc[np.isnan(C)] = (1, 1, 1, 0)
    A.plot_surface(Xs, Ys, np.where(np.isnan(C), np.nan, Z), facecolors=fc, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    A.text(-40, 0, np.nanmean(Z) + 2, name, fontsize=7, color=INK, ha='right')
A.view_init(elev=24, azim=-58); A.set_box_aspect((np.ptp(Xs), np.ptp(Ys), 260))
A.set_axis_off()
A.xaxis.pane.fill = A.yaxis.pane.fill = A.zaxis.pane.fill = False; A.grid(False)
sm = mpl.cm.ScalarMappable(cmap=SOC, norm=mpl.colors.Normalize(lv0, lv1))
cb = fig.colorbar(sm, ax=A, shrink=.45, pad=.0); cb.set_label('SOC stock per layer (Mg ha⁻¹)'); cb.outline.set_visible(False)
A.set_title('Virtual SOC stocks by layer, livestock-forest area (relief illustrative until the UAV DTM is added)', fontsize=7.5, color=INK)
fig.savefig(f'{OUT}/Fig3_3D_block_mockup.png', bbox_inches='tight'); plt.close(fig)

# ---------- Figura 4: mapa de calor P(sucesso) ----------
s = pd.read_csv(f'{RES}/curvas_resumo.csv', dtype={'cenario': str})
s['sig'] = s.cenario.replace({'observado': '0.0'}).astype(float)
NS = sorted(s.n.unique()); SG = sorted(s.sig.unique())
fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.4), sharex=True, sharey=True, constrained_layout=True)
for j, area in enumerate(['VN', 'ILP', 'IPF']):
    for i, mod in enumerate(['Ridge', 'RF']):
        A = ax[i, j]; d = s[(s.area == area) & (s.modelo == mod)].pivot(index='sig', columns='n', values='P_sucesso').reindex(index=SG, columns=NS)
        im = A.imshow(d.values, cmap=BLUE, vmin=0, vmax=1, aspect='auto', origin='lower')
        for (yy, xx), v in np.ndenumerate(d.values):
            if v >= .8: A.text(xx, yy, '●', ha='center', va='center', fontsize=4.5, color='#ffffff')
        A.set_xticks(range(len(NS))); A.set_xticklabels(NS); A.set_yticks(range(len(SG))); A.set_yticklabels([f'{v:.1f}' for v in SG])
        A.set_title(f"{AREA_NAME[area]} · {'Ridge regression' if mod == 'Ridge' else 'Random forest'}", loc='left', color=INK)
        A.axvspan(0.5, 1.5, color='none', ec=INK, lw=.8, ls='--')
        [sp.set_visible(False) for sp in A.spines.values()]; A.tick_params(length=0)
        if j == 0: A.set_ylabel('Signal strength (R²)')
        if i == 1: A.set_xlabel('Sample size (n)')
        letter(A, 'abcdef'[i * 3 + j])
cb = fig.colorbar(im, ax=ax, shrink=.7, pad=.01); cb.set_label('Probability of reaching 80% of attainable R²'); cb.outline.set_visible(False)
fig.text(0.01, -0.02, 'Dashed column: n = 20, close to the field design (20–26 points). ● probability ≥ 0.8.', fontsize=6, color=MUTED)
fig.savefig(f'{OUT}/Fig4_success_heatmap.png', bbox_inches='tight'); fig.savefig(f'{OUT}/Fig4_success_heatmap.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Figura 5: n mínimo x força do sinal ----------
nm = pd.read_csv(f'{RES}/resumo_n_minimo.csv', dtype=str)
nm['sig'] = nm.cenario.replace({'observado': '0.0'}).astype(float)
nm['nv'] = nm.n_minimo.map(lambda v: 450 if v.startswith('>') else float(v))
fig, A = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
A.axhspan(20, 26, color='#e9e9e7', zorder=0); A.text(0.02, 23, 'This study (20–26 points per area)', fontsize=6, color=MUTED, va='center', ha='left')
A.axhline(300, color=MUTED, lw=.5, ls=':')
for area in ['VN', 'ILP', 'IPF']:
    for mod, ls, mk in (('Ridge', '-', 'o'), ('RF', '--', 's')):
        d = nm[(nm.area == area) & (nm.modelo == mod)].sort_values('sig')
        A.plot(d.sig, d.nv, ls=ls, marker=mk, ms=4, lw=1.4, color=AREA_COL[area], mec='#ffffff', mew=.6)
    last = nm[(nm.area == area) & (nm.modelo == 'Ridge') & (nm.sig == 0.7)].nv.iloc[0]
A.set_yscale('log'); A.set_ylim(8, 600); A.set_yticks([10, 20, 50, 100, 200, 300, 450]); A.set_yticklabels([10, 20, 50, 100, 200, 300, 'not\nreached']); A.minorticks_off()
A.set_xlim(-0.05, 0.75); A.set_xticks([0, .3, .5, .7]); A.set_xticklabels(['Observed\n(≈0)', '0.3', '0.5', '0.7'])
A.set_xlabel('Strength of the spectral signal (R²)'); A.set_ylabel('Minimum sample size per area')
from matplotlib.lines import Line2D
h = [Line2D([], [], color=AREA_COL[a], lw=1.4, label=AREA_NAME[a]) for a in AREA_COL] + \
    [Line2D([], [], color=INK, ls='-', marker='o', ms=3, label='Ridge'), Line2D([], [], color=INK, ls='--', marker='s', ms=3, label='Random forest')]
A.legend(handles=h, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=6)
A.grid(axis='y', color='#e2e2df', lw=.5); A.set_axisbelow(True)
fig.savefig(f'{OUT}/Fig5_minimum_sample_size.png'); fig.savefig(f'{OUT}/Fig5_minimum_sample_size.pdf'); plt.close(fig)
print('ok')
