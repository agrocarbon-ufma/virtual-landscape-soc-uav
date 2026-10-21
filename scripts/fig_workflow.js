const pptxgen=require('pptxgenjs');const pres=new pptxgen();pres.defineLayout({name:'FIG',width:13.33,height:6.25});pres.layout='FIG';
pres.title='Figure 2 - Workflow';
const s=pres.addSlide();s.background={color:'FFFFFF'};
const F='Arial';
const col={field:'8C5A3C',uav:'3D7A4F',real:'5B6770',virt:'2F5D8A',exp:'7A4E8C',out:'B5651D'};
function box(id,x,y,w,h,c,title,items){
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h,rectRadius:0.08,fill:{color:'FFFFFF'},line:{color:c,width:1.75},objectName:id+'_box'});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h:0.42,rectRadius:0.08,fill:{color:c},line:{color:c,width:1.75},objectName:id+'_head'});
  s.addText(title,{x:x+0.08,y,w:w-0.16,h:0.42,fontFace:F,fontSize:12,bold:true,color:'FFFFFF',valign:'middle',margin:0,isTextBox:true,objectName:id+'_title'});
  s.addText(items.map((t,i)=>({text:t,options:{bullet:{indent:12},breakLine:i<items.length-1,paraSpaceAfter:3}})),
    {x:x+0.08,y:y+0.5,w:w-0.16,h:h-0.58,fontFace:F,fontSize:11,color:'222222',valign:'top',margin:0.02,isTextBox:true,objectName:id+'_text'});
}
function arrow(x1,y1,x2,y2,n){s.addShape(pres.shapes.LINE,{x:x1,y:y1,w:x2-x1,h:y2-y1,flipV:y2<y1,line:{color:'444444',width:1.75,endArrowType:'triangle'},objectName:n});}
const hd=(t,x,w)=>s.addText(t,{x,y:0.3,w,h:0.4,fontFace:F,fontSize:13,bold:true,color:'333333',align:'center',margin:0,isTextBox:true});
hd('Data acquisition',0.4,3.1);hd('Real-data analysis',3.95,2.95);hd('Virtual landscape',7.35,2.85);hd('Virtual experiments',10.5,2.45);
box('a',0.4,0.85,3.1,2.3,col.field,'(a) Field data',['Grid sampling, July 2024: NV 20, ICL 26, ILF 20 points','Layers 0–10, 10–20, 20–30 cm','SOC (Walkley-Black) and bulk density (core)','SOC stock = SOC × BD × thickness / 10']);
box('b',0.4,3.45,3.1,2.4,col.uav,'(b) UAV and terrain data',['MicaSense RedEdge-P, 120 m, Sept 2024','Reflectance orthomosaic resampled to 2.5 m','6 bands + 12 spectral indices','Copernicus GLO-30: elevation, slope, curvature, TPI, TWI']);
box('c',3.95,0.85,2.95,4.4,col.real,'(c) Real-data analysis',['Descriptive statistics, CV (Warrick & Nielsen)','Kruskal-Wallis + Dunn test (Bonferroni)','Sample size for the mean and power to detect differences','Spearman correlations with FDR control','Ridge and random forest, leave-one-out and repeated 5-fold CV','Result: R² ≤ 0, no spectral or terrain signal']);
box('d',7.35,0.85,2.85,4.4,col.virt,'(d) Virtual landscape',['Real indices on every pixel → normal scores → PCA','Spectral signal f: linear (PC1) or nonlinear (PC1, PC2, interaction)','z = √R²·f + √(1−R²)·ε','ε: Gaussian random field, exponential variogram fitted to field data','Anamorphosis to the observed SOC distribution','Ceiling = η² (variance explained by the signal)','Observed scenario: conditional simulation on field points']);
box('e',10.5,0.85,2.45,3.0,col.exp,'(e) Virtual experiments',['Sample size 10–300, Ridge vs RF, R² 0.3–0.7','Variogram range 25, 75, 200 m','Designs: random, grid, image-guided','Minimum signal with field n','Success: R² ≥ 0.8 × ceiling in ≥ 80% of repetitions']);
box('f',10.5,4.2,2.45,1.35,col.out,'Outputs',['Minimum sample size','Best model and design','Minimum detectable signal']);
arrow(3.5,2.0,3.95,2.0,'a_to_c');arrow(3.5,4.2,3.95,4.2,'b_to_c');
arrow(6.9,3.05,7.35,3.05,'c_to_d');arrow(10.2,2.35,10.5,2.35,'d_to_e');arrow(11.72,3.85,11.72,4.2,'e_to_f');
s.addShape(pres.shapes.LINE,{x:3.5,y:5.55,w:5.28,h:0,line:{color:'444444',width:1.75},objectName:'b_to_d_h'});s.addShape(pres.shapes.LINE,{x:8.78,y:5.25,w:0,h:0.3,line:{color:'444444',width:1.75,beginArrowType:'triangle'},objectName:'b_to_d_v'});
s.addText('Spectral indices on every pixel',{x:4.2,y:5.6,w:4.4,h:0.3,fontFace:F,fontSize:10,italic:true,color:'444444',align:'center',margin:0,isTextBox:true});

pres.writeFile({fileName:'Fig2_workflow.pptx'}).then(()=>console.log('ok'));
