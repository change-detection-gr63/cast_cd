from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import math

OUT=Path(__file__).resolve().parent
W,H=1800,3020
S=2
im=Image.new('RGB',(W*S,H*S),'white'); d=ImageDraw.Draw(im)
font='/Library/Fonts/Arial Unicode.ttf'
boldfont='/Library/Fonts/Arial Unicode.ttf'
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
diag=ET.SubElement(mx,'diagram',id='cast-cd',name='CAST-CD Architecture')
model=ET.SubElement(diag,'mxGraphModel',dx='1800',dy=str(H),grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0')
root=ET.SubElement(model,'root'); ET.SubElement(root,'mxCell',id='0'); ET.SubElement(root,'mxCell',id='1',parent='0')
n=1
boxes=[]
COL={'green':('#dcebd5','#93b77d'),'yellow':('#fff4d2','#d9b45c'),'orange':('#ffe5cc','#d89d61'),'purple':('#e9def0','#ac88be'),'blue':('#e0edf5','#82a8bf'),'gray':('#f5f5f5','#999999'),'pink':('#f4deda','#cb8e85')}
def cell(value,style,x,y,w,h):
 global n
 n+=1
 c=ET.SubElement(root,'mxCell',id=str(n),value=value,style=style,vertex='1',parent='1')
 ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
 return str(n)
def text(x,y,w,h,label,size=20,bold=False,color='#202020',align='center',emit=True):
 if emit: cell(label,f'text;html=0;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};',x,y,w,h)
 f=ImageFont.truetype(boldfont if bold else font,size*S)
 lines=label.split('\n'); lh=size*1.24*S
 cy=(y+h/2)*S-len(lines)*lh/2
 for line in lines:
  length=d.textlength(line,font=f)
  xx=(x+w/2)*S-length/2 if align=='center' else x*S
  d.text((xx,cy),line,font=f,fill=color,stroke_width=0)
  cy+=lh

def box(x,y,w,h,label='',color='gray',size=20,bold=False,dash=False):
 fill,stroke=COL[color]
 ident=cell(label,f'rounded=1;arcSize=12;whiteSpace=wrap;html=0;align=center;verticalAlign=middle;fontFamily=Arial;fontSize={size};fontStyle={1 if bold else 0};spacing=5;fillColor={fill};strokeColor={stroke};strokeWidth=1.4;'+('dashed=1;dashPattern=4 4;' if dash else ''),x,y,w,h)
 d.rounded_rectangle((x*S,y*S,(x+w)*S,(y+h)*S),radius=8*S,fill=fill,outline=stroke,width=2*S)
 if label:
  boxes.append((ident,x,y,w,h))
  text(x+5,y+3,w-10,h-6,label,size,bold,emit=False)
 return ident

def line(points,dash=False,color='#424242',arrow=True):
 global n
 n+=1
 c=ET.SubElement(root,'mxCell',id=str(n),style=f'edgeStyle=none;html=0;rounded=0;endArrow={"block" if arrow else "none"};endFill=1;strokeColor={color};strokeWidth=1.6;'+('dashed=1;dashPattern=5 4;' if dash else ''),edge='1',parent='1')
 for point,attr,pre in [(points[0],'source','exit'),(points[-1],'target','entry')]:
  px,py=point
  for ident,bx,by,bw,bh in reversed(boxes):
   if bx-1<=px<=bx+bw+1 and by-1<=py<=by+bh+1 and min(abs(px-bx),abs(px-bx-bw),abs(py-by),abs(py-by-bh))<=1:
    c.set(attr,ident)
    c.set('style',c.get('style')+f'{pre}X={(px-bx)/bw};{pre}Y={(py-by)/bh};{pre}Dx=0;{pre}Dy=0;{pre}Perimeter=0;')
    break
 g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
 for p,key in [(points[0],'sourcePoint'),(points[-1],'targetPoint')]:ET.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),**{'as':key})
 if len(points)>2:
  a=ET.SubElement(g,'Array',**{'as':'points'})
  for x,y in points[1:-1]:ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
 for (x1,y1),(x2,y2) in zip(points,points[1:]):
  if dash:
   dist=math.hypot(x2-x1,y2-y1)
   for t in range(0,int(dist),12):
    u=min(t+7,dist); d.line((S*(x1+(x2-x1)*t/dist),S*(y1+(y2-y1)*t/dist),S*(x1+(x2-x1)*u/dist),S*(y1+(y2-y1)*u/dist)),fill=color,width=2*S)
  else:d.line((x1*S,y1*S,x2*S,y2*S),fill=color,width=2*S)
 if arrow:
  x,y=points[-1];px,py=points[-2];ang=math.atan2(y-py,x-px)
  d.polygon([(x*S,y*S),((x-11*math.cos(ang)+4.5*math.sin(ang))*S,(y-11*math.sin(ang)-4.5*math.cos(ang))*S),((x-11*math.cos(ang)-4.5*math.sin(ang))*S,(y-11*math.sin(ang)+4.5*math.cos(ang))*S)],fill=color)

def panel(y,h,title,tw=620):
 cell('', 'rounded=1;arcSize=5;fillColor=none;strokeColor=#777777;strokeWidth=1;dashed=1;dashPattern=6 5;',35,y,1730,h)
 # Dashed border is kept explicit in the editable file; raster uses fine outline.
 d.rounded_rectangle((35*S,y*S,1765*S,(y+h)*S),radius=22*S,outline='#8b8b8b',width=S)
 box(60,y-15,tw,32,title,'green',21,True)

text(40,20,1720,48,'CAST-CD: Cross-Attention Spatio-Temporal Change Detection',32,True)

panel(135,670,'(a) Overall architecture',350)
# Input and shared encoder
box(65,300,145,85,'Reference I₁\nB × 3 × H × W','blue',19,True)
box(65,465,145,85,'Test I₂\nB × 3 × H × W','blue',19,True)
box(260,205,420,460,'','yellow',dash=True)
text(275,215,390,45,'Shared EfficientNet-B4 encoder',22,True)
text(275,258,390,25,'Shared weights • ImageNet pretrained',17)
for x,tag in [(290,'T₁'),(500,'T₂')]:
 for yy,lab in [(320,'Stem\n48 • H/2 × W/2'),(405,'Stage 1\n24 • H/2 × W/2'),(490,'Stage 2\n32 • H/4 × W/4'),(575,'Stage 3\n56 • H/8 × W/8')]:
  box(x,yy,150,60,lab,'pink',17)
  if yy<575:line([(x+75,yy+60),(x+75,yy+85)])
line([(210,342),(250,342),(250,350),(290,350)])
line([(210,507),(235,507),(235,300),(575,300),(575,320)])
# Fuse in vertically ordered scales
for yy,lab,c in [(405,'GDAF · C = 24\nF₁: H/2 × W/2','orange'),(490,'GDAF · C = 32\nF₂: H/4 × W/4','orange'),(575,'CTA · C = 56\nF₃: H/8 × W/8','purple')]:
 box(790,yy,210,60,lab,c,19,True)
 line([(650,yy+30),(790,yy+30)])
 line([(440,yy+48),(470,yy+48),(470,yy+72),(760,yy+72),(760,yy+45),(790,yy+45)])
# Full resolution fusion from input pair represented explicitly
box(735,175,265,80,'Concat [I₁, I₂] → Conv 3×3\n6 → 3 + PReLU + IN\nF₀: B × 3 × H × W','yellow',18)
# Direct input paths to full-resolution fusion.
line([(130,300),(130,190),(710,190),(710,200),(735,200)])
line([(65,495),(50,495),(50,165),(850,165),(850,175)])
# Decoder ascending
for yy,lab in [(575,'UpConcat 1\n56 + 32 → 64\nH/4 × W/4'),(405,'UpConcat 2\n64 + 24 → 64\nH/2 × W/2'),(175,'UpConcat 3\n64 + 3 → 32\nH × W')]:box(1100,yy,210,90,lab,'green',20,True)
line([(1000,605),(1100,605)])
line([(1000,520),(1060,520),(1060,640),(1100,640)])
line([(1000,435),(1100,435)])
line([(1205,575),(1205,495)])
line([(1205,405),(1205,265)])
line([(1000,215),(1100,215)])

# Prediction heads and auxiliary outputs
box(1370,175,175,90,'Change head → z\n1×1 + PReLU (each)\n32 → 16 → 8 → 1','gray',18,True)
box(1590,175,145,90,'Post-processing\nSigmoid > 0.5\nChange mask','blue',18,True)
line([(1310,215),(1370,215)]);line([(1545,215),(1590,215)])
box(1370,300,365,65,'Boundary head: 3×3 (32 → 16) + IN + PReLU\n→ 1×1 (16 → 1) → z_b (no final activation)','purple',17)
line([(1310,240),(1340,240),(1340,332),(1370,332)],True)
box(1370,405,365,65,'DS₂: 1×1 + PReLU (64 → 16 → 1)\nz₂: auxiliary logits at H/2 × W/2','yellow',18)
line([(1310,450),(1370,450)],True)
box(1370,575,365,65,'DS₃: 1×1 + PReLU (64 → 16 → 1)\nz₃: auxiliary logits at H/4 × W/4','yellow',18)
line([(1310,620),(1370,620)],True)
box(790,700,440,60,'E₁, E₂: original deepest encoder features\nTo contrastive loss in panel (e)','purple',18)
line([(340,635),(340,690),(850,690),(850,700)],True)
line([(610,635),(610,675),(1140,675),(1140,700)],True)
text(1260,700,475,60,'z, z_b, z₂ and z₃ feed training losses in (e).\nInference uses z → sigmoid → threshold only.',17)
text(80,775,1640,25,'Shapes: C × h × w, with H and W divisible by 8; batch B omitted. Dashed arrows in (a): auxiliary training paths.',18,color='#555555')
text(60,817,1680,42,'Fig. 1. CAST-CD fuses two temporal images at four resolutions and predicts a full-resolution binary change map.',22)

panel(900,275,'(b) Gated Difference-Attention Fusion (GDAF)',610)
box(70,960,150,70,'f₁, f₂\nB × C × h × w','blue',19,True)
box(270,960,160,70,'Difference\nD = |f₁ − f₂|','gray',19)
box(480,955,280,100,'Channel attention (SE)\nGAP → 1×1 → ReLU → 1×1 → σ\nC → C/4 → C\nA = D ⊙ SE weights(D)','orange',17)
box(830,955,280,70,'Gate: g = σ(Conv₁×₁([f₁,f₂,A]))\n3C → C','purple',17)
box(1195,955,265,70,'Gated blend\nU = g ⊙ A + (1 − g) ⊙ M','yellow',18)
box(1500,955,230,100,'Projection\nDWConv₃×₃ → IN → PReLU\n→ Conv₁×₁ (C → C)','green',16)
box(480,1080,280,50,'Temporal mean: M = (f₁ + f₂) / 2','blue',18)
line([(220,990),(270,990)])
line([(430,990),(480,990)])
line([(760,990),(830,990)])
line([(1110,990),(1195,990)])
line([(1460,990),(1500,990)])
# Original temporal features feed both the gate and mean operation.
line([(145,960),(145,940),(970,940),(970,955)])
line([(145,1030),(145,1105),(480,1105)])
# Attended difference bypasses the gate to enter the weighted blend directly.
line([(620,955),(620,922),(1325,922),(1325,955)])
line([(760,1105),(1325,1105),(1325,1025)])
text(80,1140,1640,25,'C = 24 or 32. Output: B × C × h × w.  ⊙ element-wise multiplication; σ sigmoid; GAP global average pooling.',18)

panel(1235,300,'(c) Bidirectional Cross-Temporal Channel Attention (CTA)',760)
box(70,1300,175,65,'Temporal pair\nf₁, f₂; C = 56','blue',20,True)
box(290,1285,295,95,'Direction 1 ← 2\nQ = Wq(f₁)\nK = Wk(f₂); V = Wv(f₂)','purple',19)
box(635,1285,305,95,'4 heads; 14 channels / head\nQ, K, V: B × 4 × 14 × N\nL2-normalize Q and K over N','yellow',18)
box(990,1285,345,95,'A₁₂ = softmax(τ · QK^T)\nO₁₂ = reshape(A₁₂V)\nChannel attention: 14 × 14 / head','orange',18)
box(1390,1285,335,95,'Concat [O₁₂, O₂₁]\nConv₁×₁ (112 → 56) → IN → PReLU\nF₃: B × 56 × H/8 × W/8','green',18)
for a,b in [(245,290),(585,635),(940,990),(1335,1390)]:line([(a,1330),(b,1330)])
box(290,1410,1045,60,'Direction 2 ← 1: Q = Wq(f₂), K = Wk(f₁), V = Wv(f₁). Repeat the operations above → O₂₁.','purple',19)
line([(157,1365),(157,1440),(290,1440)])
line([(1335,1440),(1557,1440),(1557,1380)])
text(70,1490,1650,28,'Wq, Wk, Wv: distinct 1×1 convolutions, reused in both directions. N = (H/8)(W/8); τ: learned temperature per head.',19)

panel(1595,300,'(d) UpConcat decoder block',420)
box(70,1650,150,60,'Input x\nPrevious scale','blue',19)
box(270,1650,210,60,'Bilinear upsample ×2\nalign_corners = True','blue',18)
box(530,1650,215,60,'Crop to skip extent\nif dimensions differ','gray',18)
box(800,1650,150,60,'Concatenate\nchannels','yellow',19)
box(1000,1650,330,60,'Depthwise Conv₃×₃\n→ PReLU → IN','green',19)
box(1380,1650,340,60,'Pointwise Conv₁×₁\n→ PReLU → IN','green',19)
box(570,1760,160,50,'Skip feature','orange',19)
box(1445,1760,275,50,'Decoder output: 64, 64, 32\nfor UpConcat 1, 2, 3','blue',18)
for left,right in [(220,270),(480,530),(745,800),(950,1000),(1330,1380)]:line([(left,1680),(right,1680)])
line([(730,1785),(875,1785),(875,1710)])
line([(650,1760),(650,1710)],True)
text(655,1720,130,25,'spatial size',16)
line([(1550,1710),(1550,1760)])
text(70,1830,1650,40,'Decoder outputs feed the prediction heads shown in (a). The heads and their targets feed the training losses shown in (e).',19)

panel(1950,1000,'(e) Training objective: predictions + targets → individual losses → weighted total',1120)
box(80,2000,320,60,'Ground-truth change mask Y\nB × H × W','gray',20,True)
text(500,2000,350,60,'Model outputs from panel (a)\nMatching symbols denote the same tensors.',18)
text(1000,2000,330,60,'Individual loss terms',21,True)
text(1480,2000,250,60,'Weighted aggregation',20,True)
# Each row has a prediction/features input, its exact target transform, and a loss.
rows=[
 (2110,'Y: original change mask','z: change logits','L_BCE\nBCEWithLogits(z, Y)','1'),
 (2230,'Y: original change mask',None,'L_FT: focal Tversky\nα = β = 0.5; γ = 0.75; smooth = 1','0.5'),
 (2350,'Boundary target\nMorphological gradient 3×3 of Y','z_b: boundary logits','L_boundary\nBCEWithLogits(z_b, boundary target)','0.3'),
 (2470,'Y₂: nearest-neighbor resize\nY → H/2 × W/2','z₂: DS₂ logits','L_DS₂\nBCEWithLogits(z₂, Y₂)','0.4'),
 (2590,'Y₄: nearest-neighbor resize\nY → H/4 × W/4','z₃: DS₃ logits','L_DS₃\nBCEWithLogits(z₃, Y₄)','0.2'),
 (2710,'Y₈: nearest-neighbor resize\nY → H/8 × W/8','E₁, E₂: deepest encoder features\nBefore CTA; 56 × H/8 × W/8','L_CFA: normalize E₁, E₂ along channels\nd = ||norm(E₁) − norm(E₂)||₂\nMean[(1 − Y₈)d² + Y₈ ReLU(2 − d)²]','0.1'),
]
box(1480,2110,250,690,'Sum of six\nweighted losses','yellow',26,True)
for yy,target,pred,loss,weight in rows:
 box(80,yy+20,320,60,target,'gray',18)
 if pred:
  box(500,yy,350,60,pred,'blue',18)
 else:
  box(500,yy,120,60,'z','blue',22)
  box(700,yy,150,60,'Sigmoid','blue',20)
  line([(620,yy+30),(700,yy+30)])
 box(1000,yy,330,90,loss,'purple',16)
 # Same ground truth branches to each target transformation.
 line([(80,2030),(55,2030),(55,yy+50),(80,yy+50)])
 line([(850,yy+30),(1000,yy+30)])
 line([(400,yy+65),(1000,yy+65)])
 line([(1330,yy+45),(1480,yy+45)])
 text(1340,yy+13,130,25,'× '+weight,18,True)
box(1480,2860,250,60,'L_total\nScalar training loss','yellow',20,True)
line([(1605,2800),(1605,2860)])
text(80,2840,1280,45,'L_total = L_BCE + 0.5 L_FT + 0.3 L_boundary + 0.4 L_DS₂ + 0.2 L_DS₃ + 0.1 L_CFA',20,True)
text(80,2890,1280,40,'BCEWithLogits receives logits. Only focal Tversky receives sigmoid probabilities; no training loss uses thresholded masks.',17)
text(60,2970,1680,30,'IN: instance normalization • Change and DS heads use PReLU after every 1×1 convolution • Boundary final convolution has no activation',17,color='#555555')
for edge in root.findall("mxCell[@edge='1']"):
 geometry=edge.find('mxGeometry')
 for pointkind,attr,pre in [('sourcePoint','source','exit'),('targetPoint','target','entry')]:
  if edge.get(attr): continue
  point=geometry.find(f"mxPoint[@as='{pointkind}']")
  px,py=float(point.get('x')),float(point.get('y'))
  for ident,bx,by,bw,bh in reversed(boxes):
   if bx-1<=px<=bx+bw+1 and by-1<=py<=by+bh+1 and min(abs(px-bx),abs(px-bx-bw),abs(py-by),abs(py-by-bh))<=1:
    edge.set(attr,ident)
    edge.set('style',edge.get('style')+f'{pre}X={(px-bx)/bw};{pre}Y={(py-by)/bh};{pre}Dx=0;{pre}Dy=0;{pre}Perimeter=0;')
    break
ET.indent(mx)
ET.ElementTree(mx).write(OUT/'CAST-CD architecture.drawio',encoding='utf-8',xml_declaration=True)
im.save(OUT/'CAST-CD architecture.png')
print(f'Created editable draw.io diagram and {W*S} × {H*S} PNG preview.')
