from flask import Flask, jsonify, request
import pandas as pd
import json

import wals
import model

app = Flask(__name__)

user_map = None
item_map = None
output_row = None
output_col = None
unique_items = None
unique_users = None
movies_map = {}

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/recommendations/<user_id>/recommendations')
def get_recommendations(user_id):
    user_rated = [item_map[i] for i, x in enumerate(user_map) if x == int(user_id)]
    print(user_rated)
    recommendations = model.generate_recommendations(int(user_id), user_rated, output_row, output_col, 6)
    recommendations = [{'movieId': int(x), 'title': movies_map[int(x)]} for x in recommendations]
    return {'recommendations': recommendations}

@app.route('/recommendations/<int:user_id>/responses', methods=['POST'])
def post_responses(user_id):
    print(request.json)
    with open("../responses/" + str(user_id), "w+") as fp:
        fp.write(json.dumps(request.json))
    return jsonify(success=True)

def initialise():
    global user_map
    global item_map
    global output_row
    global output_col
    global unique_items
    global unique_users
    user_map, item_map, train_sparse, test_sparse, unique_items, unique_users = model.clean_data("../data/ml_100k/ratings.csv")
    init_movies(unique_items)

    latent_factors = 14
    num_iters = 20

    output_row, output_col = model.train_model(train_sparse, latent_factors, num_iters)

    train_rmse = wals.get_rmse(output_row, output_col, train_sparse)
    test_rmse = wals.get_rmse(output_row, output_col, test_sparse)
    print('Train: ' + str(train_rmse) + ', Test: ' + str(test_rmse))
    
def init_movies(np_items):
    movies_df = pd.read_csv("../data/ml_100k/movies.csv", dtype={
                               'movieId': int,
                               'title': str,
                               'genres': str,
                             })
    
    i = 0
    for index, row in movies_df.iterrows():
        if (row['movieId'] in np_items):
            movies_map[i] = row['title']
            i += 1
    print(str(np_items.shape[0]) + ', ' + str(i))
    assert(np_items.shape[0] == i)

initialise()
