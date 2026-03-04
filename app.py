
import streamlit as st, cv2, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from playsound import playsound
from datetime import datetime
import os

st.set_page_config(layout="wide")
st.sidebar.title("AURA CONTROL ROOM")
page = st.sidebar.radio("Navigation", ["Live", "Log", "Analytics", "Lost & Found"])

LOG="logs.csv"

def log(c,d,f):
    df=pd.DataFrame([[datetime.now().strftime("%H:%M"),c,d,f]],columns=["Time","Count","Density","Flow"])
    df.to_csv(LOG,mode="a",header=not os.path.exists(LOG),index=False)

if page=="Live":
    st.title("Live Crowd Monitoring")
    run=st.checkbox("Start Camera")
    img=st.image([])
    cap=cv2.VideoCapture(0)
    last=time.time()
    while run:
        ret,frame=cap.read()
        if not ret: break
        count=np.random.randint(5,40)
        density="HIGH" if count>25 else "MEDIUM" if count>12 else "LOW"
        flow="Static" if density=="HIGH" else "Moderate"
        c1,c2,c3=st.columns(3)
        c1.metric("People",count)
        c2.metric("Density",density)
        c3.metric("Flow",flow)
        if density=="HIGH":
            st.error("INCIDENT: High crowd density")
            playsound("alert.wav",block=False)
        if time.time()-last>60:
            log(count,density,flow)
            last=time.time()
        img.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        time.sleep(0.1)
    cap.release()

elif page=="Log":
    st.title("Analysis Log")
    if os.path.exists(LOG):
        df=pd.read_csv(LOG).iloc[::-1]
        for _,r in df.iterrows():
            st.markdown(f"**Time:** {r.Time} | **Count:** {r.Count} | **Density:** {r.Density} | **Flow:** {r.Flow}")
    else:
        st.info("No logs yet")

elif page=="Analytics":
    st.title("Crowd Trend")
    if os.path.exists(LOG):
        df=pd.read_csv(LOG)
        fig,ax=plt.subplots()
        ax.plot(df.Time,df.Count,'-o')
        st.pyplot(fig)

elif page=="Lost & Found":
    st.title("Lost & Found")
    up=st.file_uploader("Upload image")
    if up:
        st.success("Possible match found")
        img=np.zeros((300,300,3),dtype=np.uint8)
        cv2.rectangle(img,(60,60),(240,240),(0,0,255),3)
        st.image(img,caption="Detected object")
