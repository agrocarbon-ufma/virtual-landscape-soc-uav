import glob,numpy as np,rasterio,geopandas as gpd,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8})
r=rasterio.open('data/dem/dem30_copernicus_utm23s.tif');z=r.read(1).astype(float);T=r.transform;c=T.a
def shp(n): return gpd.read_file([f for f in glob.glob('data/points/*.gpkg') if n in f][0]).to_crs(31983)
# slope (Zevenbergen-Thorne)
zp=np.pad(z,1,mode='edge')
p=(zp[1:-1,2:]-zp[1:-1,:-2])/(2*c); q=(zp[:-2,1:-1]-zp[2:,1:-1])/(2*c)
slope=np.degrees(np.arctan(np.hypot(p,q)))
# D8 accumulation
H,W=z.shape;acc=np.ones_like(z);order=np.argsort(-z,axis=None)
dirs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
for k in order:
    i,j=divmod(k,W);best=0;bt=None
    for di,dj in dirs:
        a,b=i+di,j+dj
        if 0<=a<H and 0<=b<W:
            d=(z[i,j]-z[a,b])/(c*np.hypot(di,dj))
            if d>best: best=d;bt=(a,b)
    if bt: acc[bt]+=acc[i,j]
tanb=np.tan(np.radians(np.maximum(slope,0.1)))
twi=np.log(acc*c/tanb)
sites=[('NV','VN_points'),('ICL','ILP_points'),('ILF','IPF_points')]
fig,ax=plt.subplots(3,3,figsize=(7.2,7.6),constrained_layout=True)
lab='abcdefghi'
for r_,(nm,f) in enumerate(sites):
    g=shp(f);x0,y0,x1,y1=g.total_bounds;b=400
    c0,r1=~T*(x0-b,y0-b);c1,r0=~T*(x1+b,y1+b)
    r0,r1,c0,c1=int(r0),int(r1)+1,int(c0),int(c1)+1
    ext=[T.c+c0*c,T.c+c1*c,T.f-r1*c,T.f-r0*c]
    for k,(arr,cm,t,u) in enumerate([(z,'terrain','Elevation','m'),(slope,'YlOrBr','Slope','°'),(twi,'Blues','TWI','–')]):
        a=ax[r_,k];im=a.imshow(arr[r0:r1,c0:c1],cmap=cm,extent=ext,interpolation='nearest')
        a.scatter(g.geometry.x,g.geometry.y,s=6,c='k',edgecolors='w',linewidths=0.4)
        a.set_xticks([]);a.set_yticks([]);a.set_title(f'({lab[r_*3+k]}) {nm} – {t}',fontsize=8,loc='left')
        cb=fig.colorbar(im,ax=a,shrink=0.8);cb.set_label(u,fontsize=7);cb.ax.tick_params(labelsize=6)
    ax[r_,0].plot([ext[0]+100,ext[0]+600],[ext[2]+80]*2,'k',lw=2);ax[r_,0].text(ext[0]+350,ext[2]+110,'500 m',ha='center',fontsize=6)
fig.savefig('FigS2_terrain_attributes.png',dpi=300);fig.savefig('FigS2_terrain_attributes.pdf')
