from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re
from scipy.sparse import hstack
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import ast
import pandas as pd
import gradio as gr 

movies= pd.read_csv('movies_data.csv')
'''def convert_string_to_list(input_string):
    return ast.literal_eval(input_string)

# Appliquer la fonction à la colonne 'embeddings_glove'
movies['embeddings_bag_of_words'] = movies['embeddings_bag_of_words'].apply(convert_string_to_list)

def convert_to_list(input_string):
        cleaned_string = input_string.strip()
        
        # Supprimer les crochets au début et à la fin de la chaîne
        cleaned_string = cleaned_string.lstrip('[').rstrip(']')
        
        # Séparer les éléments de la chaîne en une liste
        elements_list = [float(element) for element in cleaned_string.split()]
        
        return elements_list

# Appliquer la fonction à la colonne 'embeddings_glove'
movies['embeddings_glove'] = movies['embeddings_glove'].apply(convert_to_list)
'''

glove_file = ('glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
# Download stopwords list

stop_words = set(stopwords.words('english'))

# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]

tokenizer=StemTokenizer()
token_stop = tokenizer(' '.join(stop_words))
def extract_embeddings_from_text(text):
    vectorizer = CountVectorizer(stop_words=token_stop, tokenizer=tokenizer, max_features=1000)
    embeddings_bag_of_word = vectorizer.fit_transform([text]).toarray()
    embeddings_padded = np.zeros((1, 1000))  # Créer un vecteur de zéros de longueur 1000
    embeddings_padded[:, :embeddings_bag_of_word.shape[1]] = embeddings_bag_of_word  # Insérer les embeddings réels
    return embeddings_padded.flatten().tolist()  # Retourner un vecteur de longueur 1000
def compute_text_embedding(text, model):
    text = text.lower().split()  # Convertir le texte en une liste de mots en minuscules
    embeddings = [model[word] for word in text if word in model]  # Obtenir les embeddings pour chaque mot présent dans le modèle
    if embeddings:
        return np.mean(embeddings, axis=0).tolist()  # Calculer la moyenne des embeddings pour obtenir un seul vecteur de représentation
    else:
        return np.zeros(model.vector_size).tolist()  # Retourner un vecteur zéro si aucun mot n'est présent dans le modèle
    
import requests

def process_text(text, recommendation_type):
    if recommendation_type == 'glove':
        vector = compute_text_embedding(text,model)
        response = requests.post('http://annoy-db:5000/reco', json={'vector': vector,'type': 'glove'})
        if response.status_code == 200:
            indices= response.json()
            titles = movies['original_title'].iloc[indices].tolist()
            return titles
    
    # Make a request based on the recommendation type
    if recommendation_type == 'bagword':
        vector = extract_embeddings_from_text(text)
        response = requests.post('http://annoy-db:5000/reco', json={'vector': vector,'type': 'bagword'})
        if response.status_code == 200:
            indices= response.json()
            titles = movies['original_title'].iloc[indices].tolist()
            return titles
    else:
        return "Error in API request"

iface = gr.Interface(fn=process_text, 
                     inputs=["text", gr.Radio(["glove", "bagword"], label="Choose Recommendation Type")], 
                     outputs="text", 
                     live=True, 
                     capture_session=True)
iface.launch(server_name="0.0.0.0")

