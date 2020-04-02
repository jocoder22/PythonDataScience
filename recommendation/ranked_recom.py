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


movies = pd.read_csv("movies.csv")
reviews = pd.read_csv("reviews.csv")
del movies["Unnamed: 0"]
del reviews["Unnamed: 0"]


def movieRanks(data1, data2, col, col2):
    """The movieRanks function

    Args:
        data1 (DataFrame): the DataFrame with reviews
        data2 (DataFrame): the DataFrame with movies
        col(str): dataframe column to group by
        col2(str): dataframe column rating

    Returns:
        DataFrame: The DataFrame with ranked movies

    """
    # group the review dataset by movie_id
    result = data1.groupby(col).agg(
        {col2: ["mean", "count", "sum"], "date": "max"}
    )
    result.columns = ["_".join(col3) for col3 in result.columns.ravel()]
    result = (
        result[result["rating_count"] > 5]
        .sort_values(
            by=["rating_mean", "rating_count", "date_max"], ascending=False
        )
        .reset_index()
    )

    # merge the grouped dataframe and movie dataset
    result = result.merge(data2, on=col)

    return result


def rank_recommed(user_id, ranked, ranking):
    """The rank_recommed function will give ranked recommendation
        of movies for users

    Args:
        user_id (string): the user id for recommendataion
        ranked (int): the level of ranking

    Returns:
        list: list of ranked recommended movies

    """

    top_n = ranking.head(ranked)

    top_movies = list(top_n.movie)

    return top_movies  # a list of the n_top movies as recommended


def filter_rank_recommed(user_id, ranked, ranking, year_, genre_):
    """The rank_recommed function will give ranked recommendation
        of movies for users

    Args:
        user_id (string): the user id for recommendataion
        ranked (int): the level of ranking

    Returns:
        list: list of ranked recommended movies

    """
    ranking["genre_c"] = ranking["genre"].apply(
        lambda y: bool(set(y.split("|")) & set(genre_))
    )

    filtered = ranking[
        (ranking["year"].isin(year_)) & (ranking["genre_c"])
    ]
    filtered2 = ranking[ranking["year"].isin(year_)]

    top_n = filtered.head(ranked)

    filtered_top_movies = list(top_n.movie)

    return (
        filtered_top_movies,
        filtered2.head(ranked).movie.tolist(),
    )  


rank_df = movieRanks(reviews, movies, "movie_id", "rating")

genre_group = ["Short", "Comedy", "Adventure", "Fantasy"]
# movies['genre22'] = movies['genre'].apply(lambda x: x.split("|"))

# genre_group2 = ["Comedy", "Horror", "Adventure", "Fantasy", "Sci-Fi"]
genre_group2 = ["Drama", "Romance", "Horror", " Documentary", "Sci-Fi"]


print2(movies.head(), reviews.head())

print2(movieRanks(reviews, movies, "movie_id", "rating").head())

print2(rank_recommed("234", 9, rank_df))

# print2(filter_rank_recommed("234", 9, rank_df, ['2016',
#            '2017', '2018'], ['History']))
print2(
    filter_rank_recommed(
        "234", 9, rank_df, ["2016", "2017", "2018"], genre_group
    )
)


print2(movies.genre[:6])


print2(movies.head())
