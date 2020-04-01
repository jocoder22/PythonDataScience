import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t
from io import StringIO

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
        "timestamp": object},
    engine="python"
)

print(reviews.head())