import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t
from datetime import datetime

path = r"D:\PythonDataScience\recommendation"
os.chdir(path)

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")

# Read in the datasets
movies = pd.read_csv(
    "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat",
    sep="::",
    header=None,
    names=[
        "movie_id",
        "movie",
        "genre"],
    dtype={
        "movie_id": object,
        "movie": object,
        "genre": object},
    engine="python",
    encoding='utf-8'
)

reviews = pd.read_csv(
    "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat",
    # delimiter="::",
    sep="::",
    header=None,
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"],
    dtype={
        "movie_id": object,
        "user_id": object,
        "timestamp": int},
    parse_dates=True,
    engine="python"
)

reviews['date'] = reviews['timestamp'].apply(lambda x: pd.Timestamp(x, unit="s").date())
reviews['year'] = reviews['timestamp'].apply(lambda x: pd.Timestamp(x, unit="s").year)


dict_sol1 = {
'The number of movies in the dataset': movies.shape[0],
'The number of ratings in the dataset': reviews.shape[0],
'The number of different genres': movies['genre'].nunique(),
'The number of unique users in the dataset': reviews["user_id"].nunique(),
'The number missing ratings in the reviews dataset': reviews.rating.isna().sum(),
'The average rating given across all ratings': reviews.rating.mean(),
'The minimum rating given across all ratings': reviews.rating.min(),
'The maximum rating given across all ratings': reviews.rating.max()
}

print2(reviews.head(), movies.head(10))


# movies['century'] = movies['movie'].str.extract(r'(\d+)', expand=False)
# movies['century'] = movies['century'].apply(lambda x: x[:2] + "00")
movies['century'] = movies['movie'].apply(lambda x: x[-5:-3] + "00")
movies['year'] = movies['movie'].apply(lambda x: x[-5:-1])
print2(movies.tail(20), movies['century'].value_counts())

# and a few more below, which you can use as necessary
pd.get_dummies(movies, columns=['century']).head()


movies['genre'] = movies['genre'].astype('str')
movies['genre2'] = movies['genre'].apply(lambda x: x.split("|")[0])
movies.dropna(subset=['genre2'], inplace=True)
movies = movies[movies["genre2"] != "nan"]


pd.get_dummies(movies, columns=['genre2'])
print2(movies.head())

# remove unwanted columns
# del movies["Unnamed: 0"]
# del reviews["Unnamed: 0"]

movies.to_csv("movies.csv", index=False)
reviews.to_csv("reviews.csv", index=False)