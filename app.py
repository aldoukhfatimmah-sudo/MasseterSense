
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="MasseterSense Pro", page_icon="🦷", layout="wide")

st.markdown("""
<style>
.stApp {background:#081522;color:#e9f5fa}
[data-testid="stSidebar"] {background:#101f30}
.block-container {max-width:1450px;padding-top:2rem}
h1,h2,h3 {color:#eaf9fc!important}
[data-testid="stMetric"] {
 background:#11283b;border:1px solid #24475a;
 border-radius:16px;padding:20px
}
[data-testid="stMetricValue"] {color:#53dfd0}
div.stDownloadButton button {
 background:#087e83;color:white;border-radius:10px
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("◈ MasseterSense")
    st.caption("CLINICAL EMG MONITORING")
    st.divider()
    source = st.radio("Data Source", ["Demo Data", "Upload CSV"])
    fs = st.number_input("Sampling Rate (Hz)", 100, 20000, 2000)
    patient = st.text_input("Patient ID", "DEMO-001")
    session = st.text_input("Session", "Session 01")
    file = None
    if source == "Upload CSV":
        file = st.file_uploader("Upload EMG CSV", type="csv")
    st.divider()
    st.caption("Research prototype. Not for clinical diagnosis.")

st.title("MasseterSense | Clinical Dashboard")
st.caption("Bilateral Masseter EMG Assessment & Muscle Activity Monitoring")

if source == "Demo Data":
    t = np.arange(0, 6, 1/fs)
    rng = np.random.default_rng(42)
    env = np.where((t >= 1) & (t <= 4), 1, 0.08)
    left = env*(0.65*np.sin(2*np.pi*95*t)+0.3*np.sin(2*np.pi*170*t))
    right = env*(0.45*np.sin(2*np.pi*95*t)+0.22*np.sin(2*np.pi*170*t))
    left += rng.normal(0, 0.025, len(t))
    right += rng.normal(0, 0.025, len(t))
    df = pd.DataFrame({"time_s":t,"left_mV":left,"right_mV":right})
    st.info("Demonstration mode: synthetic EMG signals.")
else:
    if file is None:
        st.info("Upload a CSV containing left_mV and right_mV.")
        st.stop()
    df = pd.read_csv(file)
    if not {"left_mV","right_mV"}.issubset(df.columns):
        st.error("Required columns: left_mV, right_mV")
        st.stop()
    for c in ["left_mV","right_mV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf,-np.inf],np.nan).dropna(
        subset=["left_mV","right_mV"]
    )
    if "time_s" not in df:
        df["time_s"] = np.arange(len(df))/fs
    else:
        df["time_s"] = pd.to_numeric(df["time_s"],errors="coerce")
        df = df.dropna(subset=["time_s"]).sort_values("time_s")
    if len(df)<50 or df.time_s.duplicated().any():
        st.error("Invalid or insufficient synchronized data.")
        st.stop()

start = float(df.time_s.min())
end = float(df.time_s.max())
if end <= start:
    st.error("Invalid recording duration.")
    st.stop()

interval = st.slider(
    "Select EMG Analysis Interval (seconds)",
    start,end,
    (start+(end-start)*0.2,start+(end-start)*0.8),
    step=max((end-start)/2000,0.0001)
)

seg = df[(df.time_s>=interval[0]) & (df.time_s<=interval[1])]
if len(seg)<20:
    st.warning("Select a longer interval.")
    st.stop()

l = seg.left_mV.to_numpy()
r = seg.right_mV.to_numpy()
L = float(np.sqrt(np.mean(l**2)))
R = float(np.sqrt(np.mean(r**2)))
AI = 100*(L-R)/(L+R) if L+R>1e-12 else float("nan")

flags = []
for name, x in [("Left",l),("Right",r)]:
    if np.std(x)<1e-5:
        flags.append(name+": near-flat signal")
    peak = np.max(np.abs(x))
    if peak>0 and np.mean(np.abs(x)>=0.999*peak)>0.05:
        flags.append(name+": possible clipping")

a,b,c,d = st.columns(4)
a.metric("LEFT MASSETER RMS",f"{L:.3f} mV")
b.metric("RIGHT MASSETER RMS",f"{R:.3f} mV")
c.metric("ASYMMETRY INDEX",f"{AI:+.2f}%" if np.isfinite(AI) else "N/A")
d.metric("SIGNAL QUALITY","Review" if flags else "Basic checks pass")

st.divider()
col1,col2 = st.columns([1,1.4],gap="large")

with col1:
    st.subheader("Jaw Muscle Activity Map")

    # Original anatomical-style vector illustration
    jaw = """
    <svg viewBox="0 0 400 440" xmlns="http://www.w3.org/2000/svg">
    <defs>
    <linearGradient id="bone" x2="0.4" y2="1">
    <stop stop-color="#e5ecee"/>
    <stop offset="1" stop-color="#a9bbc4"/>
    </linearGradient>
    <linearGradient id="muscle" x2="1" y2="0">
    <stop stop-color="#f18b80"/>
    <stop offset="1" stop-color="#c23e56"/>
    </linearGradient>
    </defs>
    <rect width="400" height="440" rx="20" fill="#10283a"/>
    <path d="M112 338 Q88 285 90 185 Q95 52 200 48
    Q305 52 310 185 Q312 285 288 338
    Q250 400 200 404 Q150 400 112 338Z"
    fill="url(#bone)" stroke="#8fa5ae" stroke-width="3"/>
    <path d="M126 174 Q145 139 181 166 Q189 208 150 224
    Q119 221 126 174Z" fill="#607787" opacity=".7"/>
    <path d="M274 174 Q255 139 219 166 Q211 208 250 224
    Q281 221 274 174Z" fill="#607787" opacity=".7"/>
    <path d="M189 223 L180 259 Q200 279 220 259
    L211 223 L200 239Z" fill="#8499a5"/>
    <path d="M135 280 Q200 304 265 280 L258 329
    Q200 358 142 329Z" fill="#eef1ef" stroke="#9aaeb6"/>
    <path d="M142 310 Q200 326 258 310"
    stroke="#9aabb4" fill="none" stroke-width="2"/>
    <path d="M105 248 Q126 244 146 268 L151 351
    Q126 349 116 320Z"
    fill="url(#muscle)" stroke="#f4aaa1" stroke-width="3"/>
    <path d="M295 248 Q274 244 254 268 L249 351
    Q274 349 284 320Z"
    fill="url(#muscle)" stroke="#f4aaa1" stroke-width="3"/>
    <g stroke="#ffd1c8" stroke-width="2" opacity=".7">
    <path d="M113 263 L140 340 M123 261 L146 344"/>
    <path d="M287 263 L260 340 M277 261 L254 344"/>
    </g>
    <text x="45" y="415" fill="#57dfd0" font-size="15">LEFT</text>
    <text x="300" y="415" fill="#79aaff" font-size="15">RIGHT</text>
    </svg>
    """
    components.html(jaw,height=440)

    st.caption("Illustrative anatomical map — not a diagnostic image.")
    st.write("Left Muscle Activity")
    st.progress(int(100*L/max(L,R,1e-12)))
    st.write("Right Muscle Activity")
    st.progress(int(100*R/max(L,R,1e-12)))

with col2:
    st.subheader("EMG Signal Monitoring")
    view = seg.iloc[::max(1,len(seg)//3000)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view.time_s,y=view.left_mV,
        name="Left Masseter",line=dict(color="#53dfd0",width=1)
    ))
    fig.add_trace(go.Scatter(
        x=view.time_s,y=view.right_mV,
        name="Right Masseter",line=dict(color="#78aaff",width=1)
    ))
    fig.update_layout(
        template="plotly_dark",height=330,
        paper_bgcolor="#11283b",plot_bgcolor="#11283b",
        xaxis_title="Time (s)",yaxis_title="Amplitude (mV)",
        margin=dict(l=10,r=10,t=20,b=10)
    )
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Bilateral RMS Comparison")
    bar = go.Figure(go.Bar(
        x=["Left Masseter","Right Masseter"],y=[L,R],
        marker_color=["#53dfd0","#78aaff"],
        text=[f"{L:.3f}",f"{R:.3f}"],textposition="outside"
    ))
    bar.update_layout(
        template="plotly_dark",height=250,
        paper_bgcolor="#11283b",plot_bgcolor="#11283b",
        yaxis_title="RMS (mV)",showlegend=False,
        margin=dict(l=10,r=10,t=20,b=10)
    )
    st.plotly_chart(bar,use_container_width=True)

st.divider()
st.subheader("Clinical Assessment Summary")

if not np.isfinite(AI):
    summary = "Asymmetry index undefined: both channels are near zero."
elif AI>0:
    summary = f"Left masseter shows higher recorded activity. AI = {AI:+.2f}%."
elif AI<0:
    summary = f"Right masseter shows higher recorded activity. AI = {AI:+.2f}%."
else:
    summary = "Both masseters show equal recorded RMS activity."

st.info(summary+" Results are descriptive and require clinical evaluation.")

st.subheader("Signal Quality Assessment")
if flags:
    st.warning("; ".join(flags))
else:
    st.success("No basic flatline or clipping flags detected.")
st.caption("Basic checks do not establish recording quality or clinical accuracy.")

st.subheader("Generate Clinical Assessment Report")

styles = getSampleStyleSheet()
buffer = io.BytesIO()
doc = SimpleDocTemplate(buffer)
rows = [
    ["Measurement","Result"],
    ["Participant",patient],
    ["Session",session],
    ["Data Source",source],
    ["Left RMS",f"{L:.4f} mV"],
    ["Right RMS",f"{R:.4f} mV"],
    ["Asymmetry Index",f"{AI:+.2f}%" if np.isfinite(AI) else "Undefined"],
    ["Quality Flags","; ".join(flags) if flags else "No basic flags"]
]
table = Table(rows,colWidths=[180,310])
table.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),colors.lightblue),
    ("GRID",(0,0),(-1,-1),0.5,colors.grey),
    ("BOTTOMPADDING",(0,0),(-1,-1),10)
]))
story = [
    Paragraph("MasseterSense - EMG Assessment Report",styles["Title"]),
    Spacer(1,15),
    Paragraph("Research prototype - not for clinical diagnosis",styles["Normal"]),
    Spacer(1,20),table,Spacer(1,20),
    Paragraph(summary,styles["Normal"]),
    Spacer(1,12),
    Paragraph(
        "RMS computed over selected interval. "
        "AI = 100 x (L-R)/(L+R). "
        "Interpret only with synchronized, calibrated recordings.",
        styles["Normal"]
    )
]
doc.build(story)

st.download_button(
    "Download PDF Assessment",
    data=buffer.getvalue(),
    file_name="MasseterSense_Report.pdf",
    mime="application/pdf"
)
st.download_button(
    "Download EMG CSV",
    data=seg.to_csv(index=False).encode("utf-8"),
    file_name="MasseterSense_EMG.csv",
    mime="text/csv"
)

st.caption("Educational prototype | No diagnostic or treatment recommendations.")
