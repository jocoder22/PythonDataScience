import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

path = r"D:\PythonDataScience\recommendation"
os.chdir(path)


def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


movies = pd.read_csv("movies.csv")
reviews = pd.read_csv("reviews.csv")

print2(movies.head(), reviews.head())

mlistt = [73486, 75314, 68646, 99685]
user_item = reviews[reviews["movie_id"].isin(mlistt)]
user_movies = user_item.groupby(["user_id", "movie_id"])["rating"].max().unstack().dropna(axis=0)

print2(user_movies, user_item.shape, reviews.shape)
print2(user_movies.mean(axis=1).sort_values(ascending=False).head(1), user_movies.mean(axis=0).sort_values(ascending=False).iloc[0])
print2(movies[movies.movie_id == user_movies.mean(axis=0).index[0]]['movie'])


u, sigma, vt = np.linalg.svd(user_movies)
print2(u.shape, sigma.shape, vt.shape, sigma)

u_new = u[:, :len(sigma)]
sigma_new = np.diag(sigma)
vt_new = vt
print2(u_new.shape, sigma_new.shape, vt_new.shape, sigma_new)


u2, sigma2, vt2 = np.linalg.svd(user_movies, full_matrices=False)
print2("$"*8, u2.shape, sigma2.shape, vt2.shape)

assert np.allclose(np.dot(np.dot(u_new, sigma_new), vt_new), 
                    user_movies), "Something went wrong"


print2(np.dot(np.dot(u_new, sigma_new), vt_new))

total_variability = np.sum(sigma**2)

variability_comp1_comp2 = sigma[0]**2 + sigma[1]**2

percentage_explained = variability_comp1_comp2 / total_variability

print2(total_variability, variability_comp1_comp2, percentage_explained)


# for predictions
def accuracy_score(data, n):
    """The accuracy_score function calculate prediction metrics 
        for measuring model performance
 
    Args: 
        data (DataFrame): the DataFrame for recommendation
        n (int): number of hidden features for prediction 
 
    Returns: 
        sum_squared_error: sum of the squared error
        rmse: root mean square of sum of error
        accuracy: total mean of sum of error 
        mse: mean square of the sum of error
 
    """
    
    upre, sigmap, vtp = np.linalg.svd(user_movies)
    
    upred = upre[:, :n]

    sigmapred = np.diag(sigmap[:n])

    vtpred = vtp[:n, :]

    pred = np.dot(np.dot(upred, sigmapred), vtpred)

    sum_squared_error = np.sum(np.sum(data - pred) ** 2)

    accuracy = (pred - data).mean().mean()

    rmse = np.sqrt(mean_squared_error(data, pred))

    mse = mean_squared_error(data, pred)

    return sum_squared_error, rmse, accuracy, mse


print2(pred_mat(user_movies, 2))