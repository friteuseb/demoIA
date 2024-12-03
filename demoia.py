import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
import subprocess
from ultralytics import YOLO
from dotenv import load_dotenv
import plotly.express as px
import os

load_dotenv()

def main():
    st.title("Démonstrations IA pour l'Industrie")
    
    demo = st.sidebar.selectbox(
        "Choisir une démo",
        ["Assistant IA", "Analyse Prédictive", "Vision Industrielle", "YOLO Detection"]
    )
    
    if demo == "Assistant IA":
        st.header("Assistant IA pour l'Industrie")
        prompt = st.text_area("Question:", placeholder="Ex: Comment optimiser la maintenance prédictive ?")
        
        if st.button("Générer"):
            try:
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "system",
                        "content": "Expert en industrie 4.0, BIM et maintenance prédictive."
                    }, {
                        "role": "user",
                        "content": prompt
                    }]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Erreur API: {e}")



    elif demo == "Analyse Prédictive":
            st.header("Détection d'Anomalies Industrielles")
            
            # Configuration des paramètres
            n_samples = st.slider("Nombre d'échantillons", 100, 1000, 500)
            anomaly_percent = st.slider("% d'anomalies simulées", 1, 20, 5)
            
            if st.button("Simuler données capteurs"):
                # Génération des données
                dates = pd.date_range('2024-01-01', periods=n_samples, freq='h')
                
                # Paramètres normaux
                temp = np.random.normal(50, 2, n_samples)  # Température normale ~50°C
                pressure = np.random.normal(100, 5, n_samples)  # Pression normale ~100 bar
                vibration = np.random.normal(0.5, 0.1, n_samples)  # Vibration normale ~0.5g
                rotation = np.random.normal(1500, 50, n_samples)  # Vitesse rotation ~1500 rpm
                
                # Introduction d'anomalies
                n_anomalies = int(n_samples * anomaly_percent / 100)
                anomaly_idx = np.random.choice(n_samples, n_anomalies, replace=False)
                
                # Types d'anomalies
                for idx in anomaly_idx:
                    anomaly_type = np.random.choice(['temp', 'pressure', 'combined'])
                    if anomaly_type == 'temp':
                        temp[idx] += np.random.uniform(10, 20)
                    elif anomaly_type == 'pressure':
                        pressure[idx] += np.random.uniform(20, 40)
                    else:
                        temp[idx] += np.random.uniform(8, 15)
                        pressure[idx] += np.random.uniform(15, 30)
                        vibration[idx] += np.random.uniform(0.3, 0.5)
                        rotation[idx] += np.random.uniform(200, 400)
                
                # Création DataFrame
                data = pd.DataFrame({
                    'Date': dates,
                    'Temperature': temp,
                    'Pression': pressure,
                    'Vibration': vibration,
                    'Rotation': rotation
                })
                
                # Détection d'anomalies multivariées
                detector = IsolationForest(contamination=anomaly_percent/100)
                anomalies = detector.fit_predict(data[['Temperature', 'Pression', 'Vibration', 'Rotation']])
                data['Anomalie'] = anomalies
                
                # Affichage des graphiques
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Vue temporelle")
                    chart_data = data.melt(id_vars=['Date', 'Anomalie'], 
                                        value_vars=['Temperature', 'Pression', 'Vibration', 'Rotation'])
                    for var in ['Temperature', 'Pression', 'Vibration', 'Rotation']:
                        fig = px.scatter(data, x='Date', y=var, 
                                    color='Anomalie',
                                    color_discrete_map={1: 'blue', -1: 'red'},
                                    title=f"Evolution {var}")
                        st.plotly_chart(fig)
                
                with col2:
                    st.subheader("Corrélations")
                    fig = px.scatter_matrix(data,
                        dimensions=['Temperature', 'Pression', 'Vibration', 'Rotation'],
                        color='Anomalie',
                        color_discrete_map={1: 'blue', -1: 'red'})
                    st.plotly_chart(fig)
                
                # Statistiques des anomalies
                st.subheader("Analyse des anomalies")
                anomalies_df = data[data['Anomalie'] == -1]
                
                metrics = {
                    'Nombre total anomalies': len(anomalies_df),
                    '% données anormales': round(len(anomalies_df)/len(data)*100, 2),
                    'Température moyenne anomalies': round(anomalies_df['Temperature'].mean(), 2),
                    'Pression moyenne anomalies': round(anomalies_df['Pression'].mean(), 2),
                    'Vibration moyenne anomalies': round(anomalies_df['Vibration'].mean(), 2),
                    'Rotation moyenne anomalies': round(anomalies_df['Rotation'].mean(), 2)
                }
                
                col1, col2, col3 = st.columns(3)
                for i, (metric, value) in enumerate(metrics.items()):
                    with [col1, col2, col3][i % 3]:
                        st.metric(metric, value)
                
                # Sévérité des anomalies
                anomalies_df['Severite'] = anomalies_df.apply(
                    lambda row: abs(row['Temperature'] - data['Temperature'].mean()) / data['Temperature'].std() +
                            abs(row['Pression'] - data['Pression'].mean()) / data['Pression'].std() +
                            abs(row['Vibration'] - data['Vibration'].mean()) / data['Vibration'].std() +
                            abs(row['Rotation'] - data['Rotation'].mean()) / data['Rotation'].std(), axis=1
                )
                
                st.subheader("Top 5 anomalies les plus sévères")
                st.dataframe(anomalies_df.nlargest(5, 'Severite')[
                    ['Date', 'Temperature', 'Pression', 'Vibration', 'Rotation', 'Severite']
                ].style.background_gradient())

    elif demo == "Vision Industrielle":
        st.header("Détection de Visages")
        def get_cameras():
            cmd = "v4l2-ctl --list-devices"
            output = subprocess.check_output(cmd.split()).decode()
            cameras = []
            for line in output.split('\n'):
                if '/dev/video' in line:
                    cam_id = int(line.strip().replace('/dev/video', ''))
                    cameras.append((cam_id, f"Camera {cam_id}"))
            return cameras

        try:
            cameras = get_cameras()
            selected_cam = st.selectbox("Sélectionner une caméra", 
                                      options=[cam[0] for cam in cameras],
                                      format_func=lambda x: f"Camera {x}")
            
            if st.button("Démarrer caméra"):
                cap = cv2.VideoCapture(selected_cam)
                if cap.isOpened():
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    
                    frame_placeholder = st.empty()
                    stats_placeholder = st.empty()
                    stop = st.button("Arrêter")
                    
                    while not stop:
                        ret, frame = cap.read()
                        if ret:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = cascade.detectMultiScale(gray, 1.1, 4)
                            
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                                cv2.putText(frame, "Visage", (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                            
                            frame_placeholder.image(frame, channels="BGR")
                            stats_placeholder.markdown(f"Visages détectés: {len(faces)}")
                    
                    cap.release()
                else:
                    st.error("Erreur accès caméra")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

    elif demo == "YOLO Detection":
        st.header("Détection d'Objets avec YOLO")
        def get_cameras():
            cmd = "v4l2-ctl --list-devices"
            output = subprocess.check_output(cmd.split()).decode()
            cameras = []
            for line in output.split('\n'):
                if '/dev/video' in line:
                    cam_id = int(line.strip().replace('/dev/video', ''))
                    cameras.append((cam_id, f"Camera {cam_id}"))
            return cameras

        try:
            model = YOLO('yolov8n.pt')
            cameras = get_cameras()
            selected_cam = st.selectbox("Sélectionner une caméra", 
                                      options=[cam[0] for cam in cameras],
                                      format_func=lambda x: f"Camera {x}")
            
            if st.button("Démarrer caméra"):
                cap = cv2.VideoCapture(selected_cam)
                if cap.isOpened():
                    frame_placeholder = st.empty()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Détections")
                        detections_placeholder = st.empty()
                    
                    with col2:
                        st.markdown("### Statistiques")
                        stats_placeholder = st.empty()
                    
                    stop = st.button("Arrêter")
                    stats_history = []
                    
                    while not stop:
                        ret, frame = cap.read()
                        if ret:
                            results = model(frame)
                            stats = {}
                            
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    cls = int(box.cls[0])
                                    name = result.names[cls]
                                    conf = float(box.conf[0])
                                    
                                    if conf > 0.5:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                                        
                                        stats[name] = stats.get(name, 0) + 1
                            
                            frame_placeholder.image(frame, channels="BGR")
                            detections_placeholder.markdown("\n".join([f"- {k}: {v}" for k,v in stats.items()]))
                            
                            stats_history.append(stats)
                            if len(stats_history) > 10:  # Garder historique limité
                                stats_history.pop(0)
                            
                            # Créer DataFrame pour graphique
                            if stats_history:
                                df = pd.DataFrame(stats_history)
                                df.fillna(0, inplace=True)
                                stats_placeholder.line_chart(df)
                    
                    cap.release()
                else:
                    st.error("Erreur accès caméra")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()