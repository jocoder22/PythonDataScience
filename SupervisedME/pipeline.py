from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')


url2 = 'https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/data/multilabel.py'
url3 = 'https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/features/SparseInteractions.py'
# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2,
                                                               seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(
    lambda x: x[NUMERIC_COLUMNS], validate=False)


# Complete the pipeline: pl
pl = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('imputer', Imputer())
            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vectorizer', CountVectorizer())
            ]))
        ]
    )),
    ('clf', OneVsRestClassifier(LogisticRegression()))
])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)



# Edit model step in pipeline
pl = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('imputer', Imputer())
            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vectorizer', CountVectorizer())
            ]))
        ]
    )),
    ('clf', RandomForestClassifier())
])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)


# Add model step to pipeline: pl
pl = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('imputer', Imputer())
            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vectorizer', CountVectorizer())
            ]))
        ]
    )),
    ('clf', RandomForestClassifier(n_estimators=15))





    # Import pipeline
    from sklearn.pipeline import Pipeline

    # Import classifiers
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    # Import CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    # Import other preprocessing modules
    from sklearn.preprocessing import Imputer
    from sklearn.feature_selection import chi2, SelectKBest

    # Select 300 best features
    chi_k=300

    # Import functional utilities
    from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
    from sklearn.pipeline import FeatureUnion

    # Perform preprocessing
    get_text_data=FunctionTransformer(combine_text_columns, validate=False)
    get_numeric_data=FunctionTransformer(
        lambda x: x[NUMERIC_COLUMNS], validate=False)

    # Create the token pattern: TOKENS_ALPHANUMERIC
    TOKENS_ALPHANUMERIC='[A-Za-z0-9]+(?=\\s+)'

    # Instantiate pipeline: pl
    pl=Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
            ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])




    # Instantiate pipeline: pl
    pl=Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
            ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])



    # Import HashingVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer

    # Get text data: text_data
    text_data=combine_text_columns(X_train)

    # Create the token pattern: TOKENS_ALPHANUMERIC
    TOKENS_ALPHANUMERIC='[A-Za-z0-9]+(?=\\s+)'

    # Instantiate the HashingVectorizer: hashing_vec
    hashing_vec=HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

    # Fit and transform the Hashing Vectorizer
    hashed_text=hashing_vec.fit_transform(text_data)

    # Create DataFrame and print the head
    hashed_df=pd.DataFrame(hashed_text.data)
    print(hashed_df.head())



    # Import the hashing vectorizer
    from sklearn.feature_extraction.text import HashingVectorizer

    # Instantiate the winning model pipeline: pl
    pl=Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
            ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


    url4='https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/notebooks/1.0-full-model.ipynb'
