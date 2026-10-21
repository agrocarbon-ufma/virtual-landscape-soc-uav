"""Figura: paisagem real (imagem do ARP + ECOS medido) x paisagem virtual (ECOS simulado), por área.
python fig_real_virtual.py <pasta_rasters_1m> <csv_observados> <parametros_paisagem.csv> <pasta_saida>"""
import sys, os, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.enums import Resampling
from shapely import contains_xy
from scipy.stats import norm
import gstools as gs
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
warnings.filterwarnings('ignore')
RAST, OBS, PAR, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': .6, 'axes.edgecolor': '#555',
                     'axes.titlesize': 7.5, 'savefig.dpi': 400})
INK = '#222222'
SOC = LinearSegmentedColormap.from_list('soc', ['#f6efe3', '#d9b98a', '#a8743f', '#6b4423', '#3b2412'])
AREAS = {'VN': ('VN_1m.tif', '../points/VN_points.gpkg', 'Native vegetation (NV)'),
         'ILP': ('ILP_1m.tif', '../points/ILP_points.gpkg', 'Integrated crop-livestock (ICL)'),
         'IPF': ('IPF_1m.tif', '../points/IPF_points.gpkg', 'Integrated livestock-forest (ILF)')}
PIX, MARGEM, ALVO = 2.5, 15, 'solo0-30'
obs_all = pd.read_csv(OBS); par = pd.read_csv(PAR).set_index('area')
vmin, vmax = np.percentile(obs_all[ALVO], [2, 98])
fmt = FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', ' '))
rng = np.random.default_rng(2024)

def anam(z, a):
    q = np.sort(a); p = (np.arange(1, len(q) + 1) - .5) / len(q); return np.interp(norm.cdf(z), p, q)

dados = {}
for a, (tif, shp, _) in AREAS.items():
    g = gpd.read_file(f'{RAST}/{shp}').to_crs(31983)
    y = obs_all[obs_all.tratamento == a].reset_index(drop=True)[ALVO].values
    dom = g.union_all().convex_hull.buffer(MARGEM)
    with rasterio.open(f'{RAST}/{tif}') as s:
        win = rasterio.windows.from_bounds(*dom.bounds, transform=s.transform)
        f = s.res[0] / PIX; h, w = int(win.height * f), int(win.width * f)
        B, G, _, R, RE, N = s.read(window=win, out_shape=(6, h, w), resampling=Resampling.average,
                                   masked=True, boundless=True).filled(np.nan).astype(float)
        tr = s.window_transform(win) * rasterio.Affine.scale(win.width / w, win.height / h)
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = (np.asarray(v).reshape(h, w) for v in rasterio.transform.xy(tr, rows.ravel(), cols.ravel()))
    m = contains_xy(dom, xs.ravel(), ys.ravel()).reshape(h, w) & np.isfinite(B)
    rgb = np.dstack([R, G, B]); rgb = np.clip(rgb / np.nanpercentile(rgb[m], 98), 0, 1); rgb[~np.isfinite(rgb).all(2)] = 1
    rgb = np.dstack([rgb, m.astype(float)])  # fora da área fica transparente
    # paisagem virtual (cenário observado: sinal espectral ≈ 0, só o resíduo espacial)
    alc, pep = par.loc[a, 'alcance_m'], par.loc[a, 'pepita_rel']
    srf = gs.SRF(gs.Exponential(dim=2, var=1, len_scale=alc), seed=int(rng.integers(1e9)))
    c = srf((xs[m], ys[m]), mesh_type='unstructured'); c = (c - c.mean()) / c.std()
    e = np.sqrt(1 - pep) * c + np.sqrt(pep) * rng.standard_normal(m.sum())
    virt = np.full(m.shape, np.nan); virt[m] = anam((e - e.mean()) / e.std(), y)
    ext = [tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f]
    dados[a] = dict(g=g, y=y, rgb=rgb, virt=virt, ext=ext, asp=(ext[3] - ext[2]) / (ext[1] - ext[0]))

hr = [max(d['asp'], .35) for d in dados.values()]
fig, ax = plt.subplots(3, 2, figsize=(7.2, 7.2 * .5 * sum(hr) + 1.0), gridspec_kw=dict(height_ratios=hr),
                       constrained_layout=True)
for i, (a, d) in enumerate(dados.items()):
    x, yy = d['g'].geometry.x, d['g'].geometry.y
    A = ax[i, 0]; A.imshow(d['rgb'], extent=d['ext'])
    A.scatter(x, yy, c=d['y'], cmap=SOC, vmin=vmin, vmax=vmax, s=26, edgecolors='#ffffff', linewidths=.8, zorder=3)
    B_ = ax[i, 1]; im = B_.imshow(d['virt'], extent=d['ext'], cmap=SOC, vmin=vmin, vmax=vmax, interpolation='nearest')
    B_.scatter(x, yy, s=8, facecolors='none', edgecolors=INK, linewidths=.5, zorder=3)
    for j, A_ in enumerate((A, B_)):
        A_.set_xlim(d['ext'][:2]); A_.set_ylim(d['ext'][2:])
        A_.xaxis.set_major_formatter(fmt); A_.yaxis.set_major_formatter(fmt)
        A_.xaxis.set_major_locator(MaxNLocator(4)); A_.yaxis.set_major_locator(MaxNLocator(3))
        A_.tick_params(labelsize=5.5, length=2); A_.set_aspect('equal')
        if j == 1: A_.set_yticklabels([])
        lab = 'abcdef'[i * 2 + j]
        tit = (f'{AREAS[a][2]}: real landscape' if j == 0 else 'Virtual landscape (simulated SOC stock)')
        A_.set_title(f'{lab}   {tit}', loc='left', color=INK)
    ax[i, 0].set_ylabel('Northing (m)', fontsize=6)
ax[2, 0].set_xlabel('Easting (m)', fontsize=6); ax[2, 1].set_xlabel('Easting (m)', fontsize=6)
cb = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=.5, aspect=35, pad=.01)
cb.set_label('SOC stock 0–30 cm (Mg ha⁻¹)'); cb.outline.set_visible(False)
fig.suptitle('Real landscape (UAV true-colour mosaic and measured SOC at the sampling points) and virtual landscape\n'
             '(SOC simulated at 2.5 m with the observed distribution and variogram; open circles = field points). SIRGAS 2000 / UTM 23S',
             fontsize=6.5, color=INK)
for ext in ('png', 'pdf'): fig.savefig(f'{OUT}/Fig_real_vs_virtual_landscape.{ext}')
print({a: round(d['asp'], 2) for a, d in dados.items()})
