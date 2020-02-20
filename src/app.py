from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
import json
import datetime
import copy
import pickle
import requests
import random

from recommender_algo.editable_svd import EditableSVD
from explanation import InfluenceExplainer, AssociationRulesExplainer

app = Flask(__name__)

algo = None
user_algo = None
movies_map = {}
user_rated_items = None
association_rules = None
moviedb_ids = {}
api_key = ""

@app.route('/movies')
def get_movies():
    moviesList = [{'key': key, 'title': value} for key, value in movies_map.items()]
    return {'movies': moviesList}

@app.route('/movies/details/<int:movie_id>')
def get_movie_details(movie_id):
    if movie_id not in moviedb_ids:
        return jsonify(success=False)
    link_id = moviedb_ids[movie_id]
    movie_details = requests.get('https://api.themoviedb.org/3/movie/' + link_id, params={'api_key': api_key}).json()
    poster_path = movie_details['poster_path']
    description = movie_details['overview']

    release_dates = requests.get('https://api.themoviedb.org/3/movie/' + link_id + 
        '/release_dates', params={'api_key': api_key}).json()
    age_rating = get_uk_age_rating(release_dates)
    return {'poster_path': poster_path, 'description': description, 'age_rating': age_rating}

@app.route('/movies/<int:user_id>/responses', methods=['POST'])
def post_movie_responses(user_id):
    global user_rated_items
    global user_algo

    responses = request.json

    # Retrain model, and save results
    user_rated_items = {rated_movie['key']: rated_movie['rating'] for rated_movie in responses['response']}
    with open("algo-20m.pickle", "rb") as fp:
            user_algo = pickle.load(fp)
    user_algo.fit_new_user(user_id, user_rated_items)
    return jsonify(success=True)

@app.route('/recommendations/<int:user_id>/recommendations')
def get_recommendations(user_id):
    recommendations = top_recommendations(user_id)

    association_rules_explainer = AssociationRulesExplainer(user_rated_items, association_rules)
    recommendations_to_send = []
    used_recommendations = []
    for x in recommendations:
        explanation = association_rules_explainer.generate_explanation(x)
        if explanation is not None:
            movie_obj = {}
            movie_obj['movieId'] = x[0]
            movie_obj['title'] = movies_map[x[0]]
            movie_obj['explanation'] = {}
            movie_obj['explanation']['rule'] = [{'movieId': int(movie_id), 'title': movies_map[int(movie_id)]} for movie_id in explanation]
            movie_obj['explanation']['type'] = 'B'
            recommendations_to_send.append(movie_obj)
            used_recommendations.append(x[0])
            if len(recommendations_to_send) >= 5:
                break

    basic_left = 5
    influence_left = 5
    influence_explainer = InfluenceExplainer(user_id, user_rated_items, user_algo)
    for x in recommendations:
        if x[0] not in used_recommendations:
            movie_obj = {}
            movie_obj['movieId'] = int(x[0])
            movie_obj['title'] = movies_map[int(x[0])]
            if basic_left > 0:
                if influence_left > 0:
                    # rand...
                    if random.random() > 0.5:
                        # basic
                        movie_obj['explanation'] = {}
                        movie_obj['explanation']['type'] = 'A'
                        basic_left -= 1
                    else:
                        # Influence
                        positives, negatives = influence_explainer.generate_explanation(x)
                        movie_obj['explanation'] = {}
                        movie_obj['explanation']['type'] = 'C'
                        movie_obj['explanation']['positives'] = [
                            {'movieId': int(movie_id), 'title': movies_map[int(movie_id)], 'influence': influence} for movie_id, influence in positives]
                        movie_obj['explanation']['negatives'] = [
                            {'movieId': int(movie_id), 'title': movies_map[int(movie_id)], 'influence': influence} for movie_id, influence in negatives]
                        influence_left -= 1
                else:
                    # basic
                    movie_obj['explanation'] = {}
                    movie_obj['explanation']['type'] = 'A'
                    basic_left -= 1
            else :
                # Influence
                positives, negatives = influence_explainer.generate_explanation(x)
                movie_obj['explanation'] = {}
                movie_obj['explanation']['type'] = 'C'
                movie_obj['explanation']['positives'] = [
                    {'movieId': int(movie_id), 'title': movies_map[int(movie_id)], 'influence': influence} for movie_id, influence in positives]
                movie_obj['explanation']['negatives'] = [
                    {'movieId': int(movie_id), 'title': movies_map[int(movie_id)], 'influence': influence} for movie_id, influence in negatives]
                influence_left -= 1

            recommendations_to_send.append(movie_obj)
            if (basic_left == 0 and influence_left == 0):
                break
    random.shuffle(recommendations_to_send)
    print(str(recommendations_to_send))
    return {'recommendations': recommendations_to_send}

@app.route('/recommendations/<int:user_id>/responses', methods=['POST'])
def post_recommendation_responses(user_id):
    with open("../responses/" + str(user_id), "w+") as fp:
        fp.write(json.dumps(request.json))
    return jsonify(success=True)

def initialise():
    global algo
    global api_key
    with open("algo-20m.pickle", "rb") as fp:
        algo = pickle.load(fp)
    with open("../data/apiKey.txt", "r") as fp:
        api_key = fp.read().replace('\n', '')

    links_df = pd.read_csv("../data/ml-20m/links.csv", dtype={
                               'movieId': int,
                               'imdbId': str,
                               'tmdbId': str,
                             })
    for index, row in links_df.iterrows():
        if 'tmdbId' in row:
            moviedb_ids[row['movieId']] = row['tmdbId']
    init_movies()
    init_association_rules()
    
def init_movies():
    movies_df = pd.read_csv("../data/ml-20m/movies.csv", dtype={
                               'movieId': int,
                               'title': str,
                               'genres': str,
                             })
    for index, row in movies_df.iterrows():
            movies_map[row['movieId']] = row['title']

def init_association_rules():
    global association_rules
    with open("association-rules-20m.pickle", "rb") as fp:
        association_rules = pickle.load(fp)
    # Filter rules
    association_rules['consequents_length'] = association_rules['consequents'].apply(lambda x: len(x))
    association_rules['antecedents_length'] = association_rules['antecedents'].apply(lambda x: len(x))
    association_rules = association_rules[(association_rules['support'] > 0.005) & (association_rules['confidence'] > 0.3) 
        & (association_rules['antecedents_length'] < 4) & (association_rules['consequents_length'] == 1)]

def top_recommendations(user_id):
    top_n = []
    for i in movies_map:
        # Filter out rated movies
        if i not in user_rated_items:
            prediction = user_algo.predict(user_id, i)
            top_n.append((prediction.iid, prediction.est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:1000]

def get_uk_age_rating(release_dates):
    dates_arr = release_dates['results']
    for country_info in dates_arr:
        if country_info['iso_3166_1'] == 'GB':
            return country_info['release_dates'][-1]['certification']
    return 'N/A'

initialise()
