from ML2020CollaborativeFiltering.ProjectML import *
import xgboost as xgb

m = pd.read_csv("movie_titles.txt", sep=",", encoding="ISO-8859-1", header=None, names=['movieID', 'year', 'title'],
                usecols=[0, 1, 2])
r = pd.read_csv("ratings.txt", sep=",", header=None, names=['movieID', 'userID', 'rating'], usecols=[0, 1, 2])

data = pd.merge(r, m, on='movieID')
data['year'] = data['year'].round(decimals=0).astype(int)
data.duplicated(["movieID", "userID", "rating"]).sum()
data.dropna(how='all', inplace=True)

splitv = int(len(data) * 0.8)
train_data = data[:splitv]
test_data = data[splitv:]

movies_rated_per_user = train_data.groupby(by="userID")["rating"].count().sort_values(ascending=False)
ratings_per_movie = train_data.groupby(by="movieID")["rating"].count().sort_values(ascending=False)

train_sparse_data = user_item_sm(train_data)
test_sparse_data = user_item_sm(test_data)

global_average_rating = train_sparse_data.sum() / train_sparse_data.count_nonzero()

similar_user_matrix = compute_us(train_sparse_data, train_sparse_data, 100)

similar_movies = compute_msc(train_sparse_data, m.title, 2000)
print("Similar movies = {}".format(similar_movies))

train_sample_sparse_matrix = get_sample_sparse_matrix(train_sparse_data, 400, 40)
test_sparse_matrix_matrix = get_sample_sparse_matrix(test_sparse_data, 200, 20)

train_new_similar_features = create_new_similar_features(train_sample_sparse_matrix)
test_new_similar_features = create_new_similar_features(test_sparse_matrix_matrix)

x_train = train_new_similar_features.drop(["user_id", "movie_id", "rating"], axis=1)
x_test = test_new_similar_features.drop(["user_id", "movie_id", "rating"], axis=1)
y_train = train_new_similar_features["rating"]
y_test = test_new_similar_features["rating"]

clf = xgb.XGBRegressor(n_estimators=100, silent=False, n_jobs=10)
clf.fit(x_train, y_train)
y_pred_test = clf.predict(x_test)
rmse_test = error_metrics(np.nan_to_num(y_test), np.nan_to_num(y_pred_test))
print("RMSE = {}".format(rmse_test))

plot_importance(xgb, clf)
