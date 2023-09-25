import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langsmith import Client
import json 
from dotenv import load_dotenv, find_dotenv
import os

from langchain.agents.agent_toolkits.pandas.prompt import PREFIX

# Charger .env et API key
load_dotenv(find_dotenv())
openai_api_key  = os.environ['OPENAI_API_KEY']

#===============================================
#DATAFRAMES 
#===============================================

# Charger le fichier JSON
fichier = './resultats/gen2022-10-03_resultats.json'
with open(fichier, 'r') as file:
    data = json.load(file)

# 1. Créer dataframe résultats par parti
# Extraire la partie "partisPolitiques" du contenu JSON
partis_politiques = data['statistiques']['partisPolitiques']
# Convertir en DataFrame
df1 = pd.DataFrame(partis_politiques)
df1.rename(columns={
    'nbCirconscriptionsEnAvance': 'nbCirconscriptionsRemportees',
    'tauxCirconscriptionsEnAvance': 'tauxCirconscriptionsRemportees'}, inplace=True)
df1 = df1.replace({'nomPartiPolitique' : {
    "Parti libéral du Québec/Quebec Liberal Party" : "Parti libéral du Québec", 
    "Coalition avenir Québec - L'équipe François Legault" : "Coalition avenir Québec",
    "Option nationale - Pour l'indépendance du Québec" : "Option nationale",
    "Équipe Adrien Pouliot - Parti conservateur du Québec" : "Parti conservateur du Québec",
    "Parti vert du Québec/Green Party of Québec" : "Parti vert du Québec",
	"Parti conservateur du Québec - Équipe Éric Duhaime" : "Parti conservateur du Québec",
	"Bloc Montréal - Équipe Balarama Holness" : "Bloc Montréal",
	"Parti canadien du Québec / Canadian Party of Québec" : "Parti canadien du Québec"
    }})

#2. Dataframe stats générales
statistiques = data['statistiques']
statistiques.pop('partisPolitiques', None) # Remove the 'partisPolitiques' subkey
df2 = pd.DataFrame([statistiques])

# 3. dataframe résultats par circonscription
circonscriptions = data['circonscriptions']
df3 = pd.json_normalize(circonscriptions)
df3 = df3.drop('candidats', axis=1)

# ajout partis et candidats gagnants au df
partis_dict = pd.Series(df1.nomPartiPolitique.values, index=df1.abreviationPartiPolitique).to_dict()
gagnants = []

for district_data in circonscriptions:
    district_name = district_data.get('nomCirconscription', 'Inconnu')
    max_votes = -1
    winning_party_abbr = None
    winning_first_name = None
    winning_last_name = None

    for candidate in district_data.get('candidats', []):
        total_votes = candidate.get('nbVoteTotal', 0)
        party_abbr = candidate.get('abreviationPartiPolitique', 'Inconnu')
        candidate_first_name = candidate.get('prenom', 'Inconnu')
        candidate_last_name = candidate.get('nom', 'Inconnu')

        if total_votes > max_votes:
            max_votes = total_votes
            winning_party_abbr = party_abbr
            winning_first_name = candidate_first_name
            winning_last_name = candidate_last_name
	
    winning_party_full = partis_dict.get(winning_party_abbr, 'Inconnu')

    gagnants.append({
        'nomCirconscription': district_name, 
        'nomPartiPolitique': winning_party_full,
        'prenom': winning_first_name,
		'nom' : winning_last_name
    })

df_gagnants = pd.DataFrame(gagnants)
df3 = pd.merge(df3, df_gagnants, on='nomCirconscription', how='left')

# Réorganiser les colonnes
cols = df3.columns.tolist()
cols.remove('iso8601DateMAJ')
cols.remove('nomPartiPolitique')
new_cols = cols[:2] + ['nomPartiPolitique', 'prenom', 'nom'] + cols[2:-2] + ["iso8601DateMAJ"]
df3 = df3[new_cols]

# 4. dataframe résultats par candidats
candidats = data['circonscriptions']
# Convert to DataFrame including the 'candidats' key and 'nomCirconscription'
df4 = pd.json_normalize(
    circonscriptions,
    record_path='candidats',
    meta=['nomCirconscription']
)

# Ajout nom parti
df_merged = df4.merge(df1[['abreviationPartiPolitique', 'nomPartiPolitique']], on='abreviationPartiPolitique', how='left')
df_merged['nomPartiPolitique'] = df_merged['nomPartiPolitique'].fillna('').astype(str)
df_final = df_merged.groupby('abreviationPartiPolitique')['nomPartiPolitique'].apply(lambda x: ','.join(set(x))).reset_index()
df4 = df4.merge(df_final, on='abreviationPartiPolitique', how='left')
# Trouver l'index de la colonne cible
idx = df4.columns.get_loc('prenom')
cols = list(df4.columns[:idx+1]) + ['nomPartiPolitique'] + list(df4.columns[idx+1:])
cols = list(dict.fromkeys(cols))
df4 = df4[cols]

# 5. dataframe codes postaux, régions administratives et MRC 
df5 = pd.read_csv('./resultats/CP_CIRC_MUN_MRC_RA_BRUT.csv')

#==================
#CONFIG
#==================

PREFIX = """You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. 
	
	Think step by step.
	
	It is important to understand the attributes of the dataframes before working with it.

	df1 contains the results by political parties.
	df2 contains general statistics related to the election (number of votes, number of voters...).
	df3 contains statistics related to the election for each electoral district.
	df4 contains the results for each candidate.
	df5 contains information to link zip codes, municipalities or administrative regions to electoral districts.

	You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
	
	You also do not have use only the information here to answer questions - you can run intermediate queries to do exporatory data analysis to give you more information as needed.

	If the answer involves finding the name of a person, look for `nom` and `prenom` columns, and concatenate these columns to form a full name (`prenom` + `nom`)

	If the answer involves a zip code, a municipality or an administrative region, look for the dataframe df5 to find the zip code linked to the electoral district.

	Answer in French if the question is in French. 
	"""
						   
llm = ChatOpenAI(temperature=0, model="gpt-4")

agent = create_pandas_dataframe_agent(
	llm, 
	[df1,df2,df3,df4,df5], 
	agent_type=AgentType.OPENAI_FUNCTIONS,
	prefix = PREFIX,
	verbose=True, 
	max_iterations=5,
	early_stopping_method='generate',
	)

from langsmith import Client
client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


#====================
#Application Streamlit
#====================

st.set_page_config(page_title='🤖🗳️ Résultatbot, le robot des résultats électoraux')
st.title('🤖🗳️ Résultatbot, le robot des résultats électoraux')
st.info("L'IA est-elle capable de répondre à des questions en langage naturel sur des données ? Cet outil est un test avec GPT-4 et une méthode d'évaluation dévelopée par [LangChain](https://blog.langchain.dev/benchmarking-question-answering-over-csv-data/). \n\n⚠️ Cette app est en développement. Vérifiez la réponse dans les données affichées en-dessous si vous souhaitez la citer comme un fait ! \n\n💬 Cliquez sur 👎 ou 👍 pour que je puisse évaluer si ce robot fournit des réponses adéquates et l'améliorer!")

# question_list = [
# 	"Quel parti a remporté l'élection générale?",
# 	"Quelle personne a remporté le plus de voix?",
# 	"Quel a été le taux d'abstention?",
# 	"Autre question"]
# query_text = st.selectbox('Sélectionnez un exemple ou posez votre question:', question_list)
# if query_text is "Autre question":
query_text = st.text_input("Posez votre question:", placeholder = "Qui a remporté la circonscription d'Abitibi-Ouest?")

result = None
with st.form('myform', clear_on_submit=True):
	submitted = st.form_submit_button('Soumettre la question')
	if submitted:
		with st.spinner('Je cherche... 🤓'):
			response = agent({"input": query_text}, include_run_info=True)
			result = response["output"]
			run_id = response["__run"].run_id
if result is not None:
	st.info(result)
	col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
	with col_text:
		st.text("Feedback:")
	with col1:
		st.button("👍", on_click=send_feedback, args=(run_id, 1))
	with col2:
		st.button("👎", on_click=send_feedback, args=(run_id, 0))

#Afficher les dataframes
st.subheader('Résultats généraux par parti')
st.write(df1)
st.subheader('Résultats par circonscription')
st.write(df3)
st.subheader("Résultats pour l'ensemble des candidats et candidates")
st.write(df4)
st.subheader('Statistiques générales')
st.write(df2)