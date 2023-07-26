import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_data = pd.read_csv('e:\data science\Projects\movies recommendation chatbot\\netflix_titles.csv')

selected_features = ['director','country','date_added','rating','cast','duration']
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

description = movies_data['type']+' '+movies_data['director']+' '+movies_data['cast']+' '+movies_data['rating']+' '+movies_data['duration']+' '+movies_data['listed_in']+' '+movies_data['description']

titles = list(movies_data['title'])
movies = pd.DataFrame(list(zip(description, titles)), columns =['description', 'title']) 

X = movies['description']
y = movies['title']

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if len(token) >= 3]
    processed_text = " ".join(tokens)

    return processed_text

movies['preprocessed_description'] = movies['description'].apply(preprocess_text)


def recommendation(user_prompt):
    tfidf_vectorizer = TfidfVectorizer()
    movie_vectors = tfidf_vectorizer.fit_transform(movies['preprocessed_description'])
    
    preprocessed_user_prompt = preprocess_text(user_prompt)
    user_prompt_vector = tfidf_vectorizer.transform([preprocessed_user_prompt])

    similarity_scores = cosine_similarity(user_prompt_vector, movie_vectors)

    top_n = 1
    top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    recommended_movie = [(movies.iloc[i]['title'], movies_data.iloc[i]['description'], movies_data.iloc[i]['listed_in']) for i in top_indices]

    for movie in recommended_movie:
       msg = f"\nRecommended Movies:\nTitle: {movie[0]}\nDescription: {movie[1]}\nType: {movie[2]}"

    return msg

