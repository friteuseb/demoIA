# Démos IA Industrielle

Démonstrations d'applications d'IA pour l'industrie 4.0 et le BIM.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Créez un fichier `.env` avec votre clé API OpenAI :
```
OPENAI_API_KEY=votre-clé
```

## Lancement

```bash
streamlit run demoia.py
```

## Démos disponibles

### 1. Assistant IA
- Assistant spécialisé en industrie 4.0
- Questions/réponses sur maintenance prédictive, BIM, optimisation
- Utilise GPT-3.5 Turbo

### 2. Analyse Prédictive
- Simulation de données de capteurs industriels
- Détection d'anomalies multi-paramètres (température, pression, vibration, rotation)
- Visualisation temps réel et statistiques
- Configuration du nombre d'échantillons et du % d'anomalies

### 3. Vision Industrielle
- Détection de visages en temps réel
- Sélection de la webcam
- Statistiques de détection
- Visualisation des résultats

### 4. Détection YOLO
- Détection d'objets en temps réel
- 80+ classes d'objets détectables
- Affichage des probabilités
- Graphiques de suivi des détections

## Prérequis

- Python 3.10+
- Webcam
- Clé API OpenAI
- Ubuntu ou distribution Linux compatible

## Aide

Pour accéder aux caméras disponibles :
```bash
v4l2-ctl --list-devices
```
