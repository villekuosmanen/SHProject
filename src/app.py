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

from recommender_algo.editable_svd import EditableSVD

app = Flask(__name__)

algo = None
user_algo = None
movies_map = {}
user_rated_items = None
association_rules = None

@app.route('/movies')
def get_movies():
    moviesList = [{'key': key, 'title': value} for key, value in movies_map.items()]
    return {'movies': moviesList}

@app.route('/movies/<int:user_id>/responses', methods=['POST'])
def post_movie_responses(user_id):
    global user_rated_items
    global user_algo

    responses = request.json
    print(responses)

    # Retrain model, and save results
    user_rated_items = {rated_movie['key']: rated_movie['rating'] for rated_movie in responses['response']}
    with open("algo-20m.pickle", "rb") as fp:
            user_algo = pickle.load(fp)
    user_algo.fit_new_user(user_id, user_rated_items)
    return jsonify(success=True)

@app.route('/recommendations/<int:user_id>/recommendations')
def get_recommendations(user_id):
    recommendations = top_n_recommendations(user_id)

    recommendations_to_send = []
    for x in recommendations:
        movie_obj = {}
        movie_obj['movieId'] = int(x[0])
        movie_obj['title'] = movies_map[int(x[0])]
        
        # Generate explanation type A
        # rows = association_rules.loc[association_rules['consequents'].apply(lambda cons: True if int(x[0]) in cons else False)]
        # for index, row in rows.iterrows():
        #     antecedents = list(row['antecedents'])
        #     if all([x in user_rated_items.keys() for x in antecedents]):
        #         explanation = antecedents
        #         movie_obj['explanation'] = [{'movieId': int(movie_id), 'title': movies_map[int(movie_id)]} for movie_id in explanation]
        #         break

        # Generate explanation type B
        explanations = []
        for i in user_rated_items.keys():
            items_copy = user_rated_items.copy()
            items_copy.pop(i)

            user_algo.delete_user(user_id)
            user_algo.fit_new_user(user_id, items_copy)

            # Test prediction
            prediction = user_algo.predict(user_id, x[0])
            prediction_delta = x[1] - prediction.est
            explanations.append((i, prediction_delta))

        explanations.sort(key=lambda x: x[1], reverse=True)
        print(explanations)
        positives = explanations[:3]
        negatives = explanations[-3:]
        negatives.reverse()
        movie_obj['explanation'] = {}
        movie_obj['explanation']['positives'] = [
            {'movieId': int(movie_id), 'title': movies_map[int(movie_id)], 'influence': influence} for movie_id, influence in positives]
        movie_obj['explanation']['negatives'] = [
            {'movieId': int(movie_id), 'title': movies_map[int(movie_id)], 'influence': influence} for movie_id, influence in negatives]

        recommendations_to_send.append(movie_obj)
    #print(str(recommendations_to_send))
    return {'recommendations': recommendations_to_send}

@app.route('/recommendations/<int:user_id>/responses', methods=['POST'])
def post_recommendation_responses(user_id):
    with open("../responses/" + str(user_id), "w+") as fp:
        fp.write(json.dumps(request.json))
    return jsonify(success=True)

def initialise():
    global algo
    with open("algo-20m.pickle", "rb") as fp:
            algo = pickle.load(fp)
    init_movies()
    init_association_rules()
    
def init_movies():
    movies_df = pd.read_csv("../data/ml_100k/movies.csv", dtype={
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

def top_n_recommendations(user_id):
    n = 10
    top_n = []
    for i in movies_map:
        # Filter out rated movies
        if i not in user_rated_items:
            prediction = user_algo.predict(user_id, i)
            top_n.append((prediction.iid, prediction.est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:n]

initialise()
