import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import tensorflow as tf

import wals

TEST_SET_RATIO = 0.1

def clean_data(input_file):
    #Processing the data
    ratings_df = pd.read_csv(input_file, dtype={
                               'userId': np.int32,
                               'movieId': np.int32,
                               'rating': np.float32,
                               'timestamp': np.int32,
                             })
    np_users = ratings_df['userId'].values.astype(int)
    np_items = ratings_df['movieId'].values.astype(int)
    
    unique_users = np.unique(np_users)
    unique_items = np.unique(np_items)
    n_users = unique_users.shape[0]
    n_items = unique_items.shape[0]
    
    # make indexes for users and items if necessary
    max_user = unique_users[-1]
    max_item = unique_items[-1]
    if n_users != max_user or n_items != max_item:
        # make an array of 0-indexed unique user ids corresponding to the dataset
        # stack of user ids
        z = np.zeros(max_user+1, dtype=int)
        z[unique_users] = np.arange(n_users)
        u_r = z[np_users]

        # make an array of 0-indexed unique item ids corresponding to the dataset
        # stack of item ids
        z = np.zeros(max_item+1, dtype=int)
        z[unique_items] = np.arange(n_items)
        i_r = z[np_items]

        # construct the ratings set from the three stacks
        np_ratings = ratings_df['rating'].values
        ratings = np.zeros((np_ratings.shape[0], 3), dtype=object)
        ratings[:, 0] = u_r
        ratings[:, 1] = i_r
        ratings[:, 2] = np_ratings
    else:
        ratings = ratings_df.as_matrix(['user_id', 'item_id', 'rating'])
        # deal with 1-based user indices
        ratings[:, 0] -= 1
        ratings[:, 1] -= 1
        
    train_sparse, test_sparse = create_sparse_matrices(ratings, n_users, n_items)
    return ratings[:, 0], ratings[:, 1], train_sparse, test_sparse

def create_sparse_matrices(ratings, n_users, n_items):
    test_set_size = int(len(ratings) * TEST_SET_RATIO)
    test_set_idx = np.random.choice(range(len(ratings)),
                                  size=test_set_size, replace=False)
    test_set_idx = sorted(test_set_idx)

    # shift ratings into train and test sets
    ts_ratings = ratings[test_set_idx]
    tr_ratings = np.delete(ratings, test_set_idx, axis=0)

    # create training and test matrices as coo_matrix's
    u_tr, i_tr, r_tr = zip(*tr_ratings)
    train_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))

    u_ts, i_ts, r_ts = zip(*ts_ratings)
    test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_users, n_items))
    return train_sparse, test_sparse

def train_model(train_sparse, latent_factors, num_iters):
    input_tensor, row_factor, col_factor, wals_model = wals.wals_model(train_sparse, latent_factors)
    session = wals.simple_train(wals_model, input_tensor, num_iters)
    output_row = row_factor.eval(session=session)
    output_col = col_factor.eval(session=session)

    # close the trainng session now that we've evaluated the generateoutput
    session.close()
    return output_row, output_col

def generate_recommendations(user_idx, user_rated, row_factor, col_factor, k):
    """Generate recommendations for a user.
    Args:
        user_idx: the row index of the user in the ratings matrix,
        user_rated: the list of item indexes (column indexes in the ratings matrix)
            previously rated by that user (which will be excluded from the
            recommendations)
        row_factor: the row factors of the recommendation model
        col_factor: the column factors of the recommendation model
        k: number of recommendations requested
    Returns:
        list of k item indexes with the predicted highest rating, excluding
        those that the user has already rated
    """

    # bounds checking for args
    assert (row_factor.shape[0] - len(user_rated)) >= k

    # retrieve user factor
    user_f = row_factor[user_idx]

    # dot product of item factors with user factor gives predicted ratings
    pred_ratings = col_factor.dot(user_f)

    # find candidate recommended item indexes sorted by predicted rating
    k_r = k + len(user_rated)
    candidate_items = np.argsort(pred_ratings)[-k_r:]

    # remove previously rated items and take top k
    recommended_items = [i for i in candidate_items if i not in user_rated]
    recommended_items = recommended_items[-k:]

    # flip to sort highest rated first
    recommended_items.reverse()

    return recommended_items