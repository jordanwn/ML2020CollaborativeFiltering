import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


def user_item_sm(d):
    sparse_data = sparse.csr_matrix((d.rating, (d.userID, d.movieID)))
    return sparse_data


def get_average_rating(sparse_matrix, is_user):
    ax = 1 if is_user else 0
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    no_of_ratings = (sparse_matrix != 0).sum(axis=ax).A1
    rows, cols = sparse_matrix.shape
    average_ratings = {i: sum_of_ratings[i] / no_of_ratings[i] for i in range(rows if is_user else cols) if
                       no_of_ratings[i] != 0}
    return average_ratings


def compute_us(sm, train_sparse_data, limit=100):
    row_index, col_index = sm.nonzero()
    rows = np.unique(row_index)
    similar_arr = np.zeros(61700).reshape(617, 100)

    for row in rows[:limit]:
        try:

            sim = cosine_similarity(sm.getrow(row), train_sparse_data).ravel()
            similar_indices = sim.argsort()[-limit:]
            similar = sim[similar_indices]
            similar_arr[row] = similar
        except Exception as e:
            pass

    return similar_arr


def compute_msc(sm, titles_df, movieID):
    similarity = cosine_similarity(sm.T, dense_output=False)
    similar_movies = titles_df.loc[movieID - 1], similarity[movieID].count_nonzero()
    return similar_movies


def get_sample_sparse_matrix(sparseMatrix, n_users, n_movies):
    users, movies, ratings = sparse.find(sparseMatrix)
    unique_users = np.unique(users)
    unique_movies = np.unique(movies)
    np.random.seed(15)
    userS = np.random.choice(unique_users, n_users, replace=True)
    movieS = np.random.choice(unique_movies, n_movies, replace=True)
    mask = np.logical_and(np.isin(users, userS), np.isin(movies, movieS))
    sparse_sample = sparse.csr_matrix((ratings[mask], (users[mask], movies[mask])),
                                      shape=(max(userS) + 1, max(movieS) + 1))
    return sparse_sample


def create_new_similar_features(sample_sparse_matrix):
    global_avg_rating = get_average_rating(sample_sparse_matrix, False)
    global_avg_users = get_average_rating(sample_sparse_matrix, True)
    global_avg_movies = get_average_rating(sample_sparse_matrix, False)
    sample_train_users, sample_train_movies, sample_train_ratings = sparse.find(sample_sparse_matrix)
    new_features_csv_file = open("new_features.csv", mode="w")

    for user, movie, rating in zip(sample_train_users, sample_train_movies, sample_train_ratings):
        similar_arr = list()
        similar_arr.append(user)
        similar_arr.append(movie)
        similar_arr.append(sample_sparse_matrix.sum() / sample_sparse_matrix.count_nonzero())

        similar_users = cosine_similarity(sample_sparse_matrix[user], sample_sparse_matrix).ravel()
        indices = np.argsort(-similar_users)[1:]
        ratings = sample_sparse_matrix[indices, movie].toarray().ravel()
        top_similar_user_ratings = list(ratings[ratings != 0][:5])
        top_similar_user_ratings.extend([global_avg_rating[movie]] * (5 - len(ratings)))
        similar_arr.extend(top_similar_user_ratings)

        similar_movies = cosine_similarity(sample_sparse_matrix[:, movie].T, sample_sparse_matrix.T).ravel()
        similar_movies_indices = np.argsort(-similar_movies)[1:]
        similar_movies_ratings = sample_sparse_matrix[user, similar_movies_indices].toarray().ravel()
        top_similar_movie_ratings = list(similar_movies_ratings[similar_movies_ratings != 0][:5])
        top_similar_movie_ratings.extend([global_avg_users[user]] * (5 - len(top_similar_movie_ratings)))
        similar_arr.extend(top_similar_movie_ratings)

        similar_arr.append(global_avg_users[user])
        similar_arr.append(global_avg_movies[movie])
        similar_arr.append(rating)

        new_features_csv_file.write(",".join(map(str, similar_arr)))
        new_features_csv_file.write("\n")

    new_features_csv_file.close()
    new_features_df = pd.read_csv('new_features.csv',
                                  names=["user_id", "movie_id", "gloabl_average", "similar_user_rating1",
                                         "similar_user_rating2", "similar_user_rating3",
                                         "similar_user_rating4", "similar_user_rating5",
                                         "similar_movie_rating1", "similar_movie_rating2",
                                         "similar_movie_rating3", "similar_movie_rating4",
                                         "similar_movie_rating5", "user_average",
                                         "movie_average", "rating"])
    return new_features_df


def error_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def plot_importance(model, clf):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    model.plot_importance(clf, ax=ax, height=0.3)
    plt.xlabel("F Score", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    plt.title("Feature Importance", fontsize=20)
    plt.tick_params(labelsize=15)

    plt.show()
