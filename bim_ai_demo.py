import streamlit as st
from openai import OpenAI
import numpy as np
import json
import time
from dotenv import load_dotenv
import os
import plotly.graph_objects as go

load_dotenv()

def handle_model_generation(building_desc, constraints):
    try:
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Expert en génération de paramètres BIM. Répondre uniquement en format JSON valide."},
                {"role": "user", "content": f"""
                Générer un JSON avec les paramètres BIM suivants:
                - dimensions du bâtiment
                - liste des composants principaux
                - matériaux utilisés
                - caractéristiques techniques
                
                Pour ce bâtiment:
                Description: {building_desc}
                Contraintes: {constraints}
                """}
            ]
        )
        
        # Récupération du contenu de la réponse
        content = response.choices[0].message.content.strip()
        
        # Vérification si le contenu commence et finit par des accolades
        if not (content.startswith('{') and content.endswith('}')):
            # Si ce n'est pas le cas, on essaie de trouver le JSON dans le texte
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                content = content[start:end]
            else:
                raise ValueError("Pas de JSON valide dans la réponse")

        # Parse du JSON
        params = json.loads(content)
        return params

    except json.JSONDecodeError as e:
        st.error(f"Erreur de format JSON: {str(e)}")
        st.text("Réponse brute reçue:")
        st.text(response.choices[0].message.content)
        return None
    except Exception as e:
        st.error(f"Erreur lors de la génération: {str(e)}")
        return None

def handle_optimization(file, goals):
    recommendations = {
        "Performance énergétique": [
            "Ajustement orientation fenêtres sud +15°",
            "Augmentation épaisseur isolation nord",
            "Ajout brise-soleil façade ouest"
        ],
        "Impact environnemental": [
            "Substitution béton par alternative bas carbone",
            "Integration matériaux biosourcés en façade",
            "Optimisation structure pour -20% matière"
        ],
        "Coût construction": [
            "Optimisation des quantités de matériaux",
            "Standardisation des éléments préfabriqués",
            "Réduction des coûts logistiques"
        ],
        "Confort thermique": [
            "Amélioration de l'isolation thermique",
            "Optimisation de la ventilation naturelle",
            "Installation de protections solaires adaptatives"
        ],
        "Luminosité naturelle": [
            "Redimensionnement des ouvertures",
            "Ajout de puits de lumière",
            "Optimisation des réflexions lumineuses"
        ]
    }
    return recommendations

def handle_conversion(input_format, output_format, file, options):
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)  # Accéléré pour la démo
        progress.progress(i + 1)
    return True


def extract_numeric_value(value):
    """Extrait la valeur numérique d'une chaîne (ex: '12m' -> 12)"""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(''.join(filter(lambda x: x.isdigit() or x == '.', str(value))))
    except:
        return None

import streamlit as st
from openai import OpenAI
import numpy as np
import json
import time
from dotenv import load_dotenv
import os
import plotly.graph_objects as go

def extract_numeric_value(value):
    """Extrait la valeur numérique d'une chaîne (ex: '12m' -> 12)"""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(''.join(filter(lambda x: x.isdigit() or x == '.', str(value))))
    except:
        return None

def create_building_visualization(params):
    try:
        # Extraction des dimensions avec gestion d'erreurs
        dimensions = params.get('dimensions', {})
        
        # Récupération des valeurs avec gestion des différents formats possibles
        height = None
        if 'hauteur_max' in dimensions:
            height = extract_numeric_value(dimensions['hauteur_max'])
        elif 'hauteur' in dimensions:
            height = extract_numeric_value(dimensions['hauteur'])
        if height is None:
            height = 12  # Valeur par défaut
            
        surface = None
        if 'surface_au_sol_max' in dimensions:
            surface = extract_numeric_value(dimensions['surface_au_sol_max'])
        elif 'surface_au_sol' in dimensions:
            surface = extract_numeric_value(dimensions['surface_au_sol'])
        if surface is None:
            surface = 400  # Valeur par défaut
            
        floors = None
        if 'nombre_etages' in dimensions:
            floors = extract_numeric_value(dimensions['nombre_etages'])
        if floors is None:
            floors = 3  # Valeur par défaut

        # Calcul des dimensions du bâtiment
        width = np.sqrt(surface)
        length = width
        floor_height = height / floors
        
        # Création des points pour le bâtiment
        x = np.array([0, width, width, 0, 0, 0, width, width, 0])
        y = np.array([0, 0, length, length, 0, 0, 0, length, length])
        z = np.array([0, 0, 0, 0, 0, height, height, height, height])
        
        # Création de la figure
        fig = go.Figure()
        
        # Ajout du bâtiment principal
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0, 0, 0, 0, 4, 4, 4, 4, 0, 1, 2, 3],
            j=[1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8],
            k=[2, 3, 4, 1, 6, 7, 8, 5, 1, 2, 3, 4],
            opacity=0.6,
            color='lightblue',
            name='Structure'
        ))
        
        # Ajout des étages
        for floor in range(int(floors)):
            z_level = floor * floor_height
            fig.add_trace(go.Scatter3d(
                x=[0, width, width, 0, 0],
                y=[0, 0, length, length, 0],
                z=[z_level] * 5,
                mode='lines',
                line=dict(color='blue', width=1),
                name=f'Étage {floor+1}' if floor > 0 else 'Rez-de-chaussée'
            ))
        
        # Extraction des composants
        composants = params.get('composants_principaux', [])
        # Conversion en liste si nécessaire (gestion du format dict)
        if isinstance(composants, dict):
            composants = list(composants.values())
        
        # Si le bâtiment a un toit végétalisé
        if any('végétalis' in str(comp).lower() for comp in composants):
            fig.add_trace(go.Scatter3d(
                x=[0, width, width, 0, 0],
                y=[0, 0, length, length, 0],
                z=[height] * 5,
                mode='lines+markers',
                line=dict(color='green', width=4),
                name='Toit végétalisé'
            ))
        
        # Si le bâtiment a des panneaux solaires
        if any('solaire' in str(comp).lower() for comp in composants):
            panel_points = np.linspace(0, width, 5)
            for i, x in enumerate(panel_points[:-1]):
                fig.add_trace(go.Scatter3d(
                    x=[x, x+width/5],
                    y=[length/4, length/4],
                    z=[height+0.2, height+0.2],
                    mode='lines',
                    line=dict(color='darkblue', width=3),
                    name='Panneau solaire' if i == 0 else None,
                    showlegend=(i == 0)
                ))
        
        # Configuration de la mise en page
        fig.update_layout(
            title="Visualisation 3D du bâtiment",
            scene=dict(
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis_title="Largeur (m)",
                yaxis_title="Longueur (m)",
                zaxis_title="Hauteur (m)"
            ),
            showlegend=True,
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la visualisation: {str(e)}")
        st.write("Structure des paramètres reçus:", params)  # Debug
        return None



def bim_ai_demo():
    if not os.getenv('OPENAI_API_KEY'):
        st.error("Clé API OpenAI non trouvée. Veuillez vérifier votre fichier .env")
        return
        
    st.header("IA Générative pour BIM/CAO")
    
    demo_type = st.selectbox(
        "Type de génération",
        ["Modèle 3D par prompt", "Optimisation design", "Conversion format"]
    )
    
    if demo_type == "Modèle 3D par prompt":
        st.subheader("Génération de modèle 3D par description")
        
        building_desc = st.text_area(
            "Description du bâtiment",
            "Bâtiment moderne de 3 étages avec façade vitrée, toit terrasse végétalisé et panneaux solaires"
        )
        
        constraints = st.text_area(
            "Contraintes techniques",
            "Surface au sol max 400m2, hauteur max 12m, orientation sud-est"
        )
        
        if st.button("Générer modèle"):
                with st.spinner('Génération en cours...'):
                    params = handle_model_generation(building_desc, constraints)
                    if params:
                        st.success("Paramètres générés avec succès!")
                        st.json(params)
                        
                        # Création et affichage de la visualisation 3D
                        fig = create_building_visualization(params)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Ajout d'instructions pour l'interaction
                            st.info("💡 Vous pouvez interagir avec le modèle 3D :\n"
                                "- Faire tourner : Cliquer-glisser\n"
                                "- Zoomer : Molette de la souris\n"
                                "- Déplacer : Clic-droit + glisser")
                
    elif demo_type == "Optimisation design":
        st.subheader("Optimisation de design par IA")
        
        uploaded_file = st.file_uploader("Charger modèle IFC/BIM", type=['ifc'])
        
        optimization_goals = st.multiselect(
            "Objectifs d'optimisation",
            ["Performance énergétique", "Coût construction", "Impact environnemental", 
             "Confort thermique", "Luminosité naturelle"]
        )
        
        if uploaded_file and optimization_goals and st.button("Optimiser"):
            with st.spinner('Analyse en cours...'):
                st.info("Analyse du modèle et génération de recommandations...")
                
                recommendations = handle_optimization(uploaded_file, optimization_goals)
                
                st.subheader("Recommandations d'optimisation")
                for goal in optimization_goals:
                    if goal in recommendations:
                        st.write(f"**{goal}:**")
                        for rec in recommendations[goal]:
                            st.write(f"- {rec}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Réduction consommation", "-25%")
                col2.metric("Réduction coûts", "-18%")
                col3.metric("Réduction CO2", "-30%")
            
    elif demo_type == "Conversion format":
        st.subheader("Export et conversion de formats")
        
        input_format = st.selectbox("Format source", ["IFC", "RVT", "SKP", "DWG"])
        output_format = st.selectbox("Format cible", ["IFC", "OBJ", "FBX", "GLTF"])
        
        preserve_options = st.multiselect(
            "Éléments à préserver",
            ["Géométrie", "Matériaux", "Métadonnées", "Hiérarchie", "Textures"]
        )
        
        uploaded_file = st.file_uploader(f"Charger fichier {input_format}", type=[input_format.lower()])
        
        if uploaded_file and st.button("Convertir"):
            with st.spinner(f'Conversion {input_format} vers {output_format} en cours...'):
                st.info(f"Conversion {input_format} vers {output_format}...")
                
                if handle_conversion(input_format, output_format, uploaded_file, preserve_options):
                    st.success(f"Conversion réussie! Préservation de: {', '.join(preserve_options)}")
                    st.download_button(
                        "Télécharger résultat",
                        b"Fichier converti simule",
                        file_name=f"converti.{output_format.lower()}"
                    )

if __name__ == "__main__":
    bim_ai_demo()