import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title='MasseterSense',page_icon='🦷',layout='wide')
st.title('MasseterSense | Bilateral Masseter EMG')
st.caption('Research prototype only — not a diagnostic or treatment recommendation tool.')

source=st.radio('Data source',['Demo data','Upload CSV'],horizontal=True)
fs=st.number_input('Sampling frequency (Hz)',min_value=100,max_value=20000,value=2000,step=100)
if source=='Demo data':
    t=np.arange(0,6,1/fs)
    rng=np.random.default_rng(42)
    env=np.where((t>=1)&(t<=4),1.0,0.12)
    left=env*(0.65*np.sin(2*np.pi*95*t)+0.3*np.sin(2*np.pi*170*t))+rng.normal(0,0.035,len(t))
    right=env*(0.47*np.sin(2*np.pi*95*t)+0.24*np.sin(2*np.pi*170*t))+rng.normal(0,0.035,len(t))
    df=pd.DataFrame({'time_s':t,'left_mV':left,'right_mV':right})
    st.info('DEMO: Synthetic signals only. Not human measurements.')
else:
    uploaded=st.file_uploader('CSV columns: left_mV, right_mV; optional time_s',type='csv')
    if uploaded is None:
        st.stop()
    try:
        df=pd.read_csv(uploaded)
        if not {'left_mV','right_mV'}.issubset(df.columns):
            st.error('CSV must contain left_mV and right_mV columns.')
            st.stop()
        df=df[['left_mV','right_mV']+(['time_s'] if 'time_s' in df.columns else [])].copy()
        df[['left_mV','right_mV']]=df[['left_mV','right_mV']].apply(pd.to_numeric,errors='coerce')
        df=df.replace([np.inf,-np.inf],np.nan).dropna()
        if 'time_s' not in df.columns:
            df['time_s']=np.arange(len(df))/fs
        else:
            df['time_s']=pd.to_numeric(df['time_s'],errors='coerce')
            df=df.dropna().sort_values('time_s')
        if len(df)<20:
            st.error('Need at least 20 valid samples.')
            st.stop()
    except Exception as e:
        st.error(f'Could not read CSV: {e}')
        st.stop()
    st.warning('Only use de-identified, authorized data. Confirm both channels share a synchronized time base and units are mV.')

st.subheader('Select clenching interval')
start=float(df.time_s.min()); end=float(df.time_s.max())
if end<=start:
    st.error('Invalid time range.');st.stop()
interval=st.slider('Time interval (seconds)',min_value=start,max_value=end,value=(max(start,min(end,start+1)),max(start,min(end,start+4))),step=max((end-start)/1000,0.001))
seg=df[(df.time_s>=interval[0])&(df.time_s<=interval[1])]
if len(seg)<10:
    st.warning('Select a longer interval.');st.stop()
L=float(np.sqrt(np.mean(np.square(seg.left_mV.to_numpy()))))
R=float(np.sqrt(np.mean(np.square(seg.right_mV.to_numpy()))))
ai=100*(L-R)/(L+R) if L+R>1e-10 else np.nan

c1,c2,c3=st.columns(3)
c1.metric('Left RMS',f'{L:.4f} mV')
c2.metric('Right RMS',f'{R:.4f} mV')
c3.metric('Asymmetry Index',f'{ai:+.2f}%' if np.isfinite(ai) else 'Undefined')

st.subheader('Muscle activity map (illustrative)')
fig,ax=plt.subplots(figsize=(5,3.5))
from matplotlib.patches import Ellipse, Circle, Arc
ax.add_patch(Ellipse((0,0),2.6,3.2,facecolor='#e8d6c3',edgecolor='#76675c',lw=2))
ax.add_patch(Ellipse((-0.8,-0.3),0.55,1.05,facecolor=plt.cm.Reds(min(L/2,1)),edgecolor='#9b5555',alpha=.85))
ax.add_patch(Ellipse((0.8,-0.3),0.55,1.05,facecolor=plt.cm.Reds(min(R/2,1)),edgecolor='#9b5555',alpha=.85))
for x in (-0.45,0.45):ax.add_patch(Circle((x,0.5),0.07,color='#4e3e35'))
ax.add_patch(Arc((0,-0.75),0.7,0.25,theta1=190,theta2=350,color='#4e3e35'))
ax.text(-0.8,-1.75,'LEFT',ha='center');ax.text(0.8,-1.75,'RIGHT',ha='center')
ax.set_xlim(-1.6,1.6);ax.set_ylim(-1.9,1.8);ax.axis('off')
st.pyplot(fig)
plt.close(fig)
st.caption('Schematic, not anatomical electrode placement. Colors indicate relative measured amplitude, not pathology.')

st.subheader('EMG waveforms')
chart=seg[['time_s','left_mV','right_mV']].set_index('time_s')
st.line_chart(chart)

st.subheader('Basic signal-quality flags')
flags=[]
for label,col in [('Left','left_mV'),('Right','right_mV')]:
    x=seg[col].to_numpy()
    if np.std(x)<1e-5: flags.append(f'{label}: near-flat signal')
    if np.max(np.abs(x))>0 and np.mean(np.abs(x)>=0.999*np.max(np.abs(x)))>.05:flags.append(f'{label}: possible clipping; inspect raw waveform')
    if np.any(~np.isfinite(x)):flags.append(f'{label}: non-finite values')
if flags: st.warning('; '.join(flags))
else: st.success('No basic flags detected. This does NOT validate signal quality or clinical accuracy.')
st.caption('This prototype does not automatically remove noise, calibrate channel gains, or establish clinical thresholds. Use identically calibrated channels and appropriate processing before interpretation.')

st.subheader('Download report')
styles=getSampleStyleSheet()
buf=io.BytesIO()
doc=SimpleDocTemplate(buf)
summary='Equal RMS activity' if np.isfinite(ai) and abs(ai)<1e-8 else ('Higher left recorded RMS' if ai>0 else 'Higher right recorded RMS') if np.isfinite(ai) else 'Index undefined'
story=[Paragraph('MasseterSense — Research Measurement Report',styles['Title']),Spacer(1,12),Paragraph('Prototype / Not for clinical diagnosis',styles['Normal']),Spacer(1,12)]
rows=[['Measure','Value'],['Source',source],['Interval (s)',f'{interval[0]:.2f}–{interval[1]:.2f}'],['Left RMS',f'{L:.4f} mV'],['Right RMS',f'{R:.4f} mV'],['Asymmetry Index',f'{ai:+.2f}%' if np.isfinite(ai) else 'Undefined'],['Interpretation',summary],['Quality flags','; '.join(flags) if flags else 'No basic flags detected']]
table=Table(rows,colWidths=[170,300]);table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
story.append(table);story.append(Spacer(1,15));story.append(Paragraph('RMS is computed over the selected interval. AI = 100 × (L−R)/(L+R). Findings are descriptive and require signal validation and clinical assessment.',styles['Normal']))
doc.build(story)
st.download_button('Download PDF report',buf.getvalue(),file_name='MasseterSense_report.pdf',mime='application/pdf')
st.download_button('Download example CSV',df.to_csv(index=False).encode('utf-8'),file_name='MasseterSense_data.csv',mime='text/csv')
