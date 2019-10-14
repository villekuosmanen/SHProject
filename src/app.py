from flask import Flask

import wals
import model

app = Flask(__name__)

user_map = None
item_map = None
output_row = None
output_col = None

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/recommendations/<user_id>')
def get_recommendations(user_id):
    user_rated = [item_map[i] for i, x in enumerate(user_map) if x == int(user_id)]
    print(user_rated)
    recommendations = model.generate_recommendations(int(user_id), user_rated, output_row, output_col, 6)
    recommendations = [int(x) for x in recommendations]
    return {'recommendations': recommendations}

def initialise():
    global user_map
    global item_map
    global output_row
    global output_col
    user_map, item_map, train_sparse, test_sparse = model.clean_data("../data/ratings.csv")
    latent_factors = 14
    num_iters = 20

    output_row, output_col = model.train_model(train_sparse, latent_factors, num_iters)

    train_rmse = wals.get_rmse(output_row, output_col, train_sparse)
    test_rmse = wals.get_rmse(output_row, output_col, test_sparse)
    print('Train: ' + str(train_rmse) + ', Test: ' + str(test_rmse))

initialise()
