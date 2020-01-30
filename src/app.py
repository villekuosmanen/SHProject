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

from recommender_algo.editable_svd import EditableSVD

app = Flask(__name__)

algo = None
movies_map = {}
user_rated_items = None

@app.route('/movies')
def get_movies():
    moviesList = [{'key': key, 'title': value} for key, value in movies_map.items()]
    return {'movies': moviesList}

@app.route('/movies/<int:user_id>/responses', methods=['POST'])
def post_movie_responses(user_id):
    global user_rated_items

    responses = request.json
    print(responses)

    # Retrain model, and save results
    user_rated_items = {rated_movie['key']: rated_movie['rating'] for rated_movie in responses['response']}
    algo.fit_new_user(user_id, user_rated_items)
    return jsonify(success=True)

@app.route('/recommendations/<int:user_id>/recommendations')
def get_recommendations(user_id):
    recommendations = top_n_recommendations(user_id)
    recommendations = [{'movieId': int(x[0]), 'title': movies_map[int(x[0])]} for x in recommendations]
    return {'recommendations': recommendations}

@app.route('/recommendations/<int:user_id>/responses', methods=['POST'])
def post_recommendation_responses(user_id):
    with open("../responses/" + str(user_id), "w+") as fp:
        fp.write(json.dumps(request.json))
    return jsonify(success=True)

def initialise():
    global algo

    dev_file = "../data/ml_100k/ratings.csv"
    prod_file = "../data/ml-20m/ratings.csv"
    ratings_df = pd.read_csv(dev_file, dtype={
        'userId': np.int32,
        'movieId': np.int32,
        'rating': np.float32,
        'timestamp': np.int32,
    })

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=.25)
    algo = EditableSVD()

    train_start_time = datetime.datetime.now()
    algo.fit(trainset)
    train_end_time = datetime.datetime.now()
    print("Training duration: " + str(train_end_time - train_start_time))

    init_movies()
    
def init_movies():
    movies_df = pd.read_csv("../data/ml_100k/movies.csv", dtype={
                               'movieId': int,
                               'title': str,
                               'genres': str,
                             })
    
    for index, row in movies_df.iterrows():
            movies_map[row['movieId']] = row['title']

def top_n_recommendations(user_id):
    n = 6
    top_n = []
    for i in movies_map:
        # Filter out rated movies
        if i not in user_rated_items:
            prediction = algo.predict(user_id, i)
            top_n.append((prediction.iid, prediction.est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    top_n.sort(key=lambda x: x[1], reverse=True)
    print(str(user_id) + ": " + str(top_n))
    return top_n[:n]

initialise()
