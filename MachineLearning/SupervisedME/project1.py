from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
plt.style.use('ggplot')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'

columnsName = ['party', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze',
               'el-salvador-aid', 'religious-groups-in-schools', 'satellite',
               'aid-to-nicaraguan-contras', 'missile', 'immigration',
               'synfuels-corporation-cutback', 'education', 'superfund-right-to-sue',
               'crime', 'duty-free-exports', 'exportafrica']


df = pd.read_csv(url, names=columnsName, sep=',')


df[df == '?'] = -1000
df[df == 'y'] = 1
df[df == 'n'] = 0
df.party = df.party.astype('category')
df.party = pd.get_dummies(df['party'])
df[df.columns[1:]] = df.drop('party', axis=1).astype('float')
# [x for x in X_train if X_train[x].dtype != 'object']

# dummy_label = pd.get_dummies(df[LABELS])
NON_LABELS = [c for c in df.columns if c != 'party']
print(NON_LABELS)
print(df.head())
print(df.dtypes)

X_train, X_test, y_train, y_test = train_test_split(df[df.columns[1:]],
                                                    df.party,
                                                    test_size=0.2, random_state=42)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))


# # Load the holdout data: holdout
# holdout = pd.read_csv('HoldoutData.csv', index_col=0)

# # Generate predictions: predictions
# predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))


# # Format predictions in DataFrame: prediction_df
# prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
#                              index=holdout.index,
#                              data=predictions)


# # Save prediction_df to csv
# prediction_df.to_csv('predictions.csv')

# # Submit the predictions for scoring: score
# score = score_submission(pred_path='predictions.csv')

# # Print score
# print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))



extra = ["This is my boy In this exercise, you'll study"," " ,"the effects of tokenizing in different", 
         "ways by comparing the bag-of-words",
        "representations resulting from different token patterns.",
         "You will focus on one feature only, the"," Position_Extra column, ",
         "which describes any additional information", " ","not captured by the Position_Type label.",
         "For example", " ", "in the", " Shell you", " ", "can check", "out the budget item ",
         "in row 8960 of the data using df.loc[8960]. ","Looking at the output reveals that this Object_Description ",
         "is overtime pay. For who? The ","Position Type is merely other", "but the Position Extra elaborates:","BUS DRIVER",
         "Explore the column further to ", "see more instances. It has a lot of", "NaN values.",
         "Your task is to turn the"," raw text in this"," column into a bag-of-words representation by creating", 
         "tokens that contain only alphanumeric characters.",
         "For comparison purposes, the","first 15 tokens of vec_basic, which splits df.Position_Extra into tokens", 
         "when it encounters only whitespace characters", "have been printed along with","the length of the representation."]


df['Position_Extra'] = pd.Series(extra)
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Fill missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace=True)

# Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])


def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    # Replace nans with blanks
    text_data.fillna("", inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)






# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1, 2))

# Create the text vector
text_vector = combine_text_columns(df)

# Fit and transform vec_basic
vec_basic.fit_transform(text_vector)


# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(
    len(vec_basic.get_feature_names())))

# Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector)

# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))
print(df.head())
print(df.columns)


# Split and select numeric data only, no nans
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],
                                                    pd.get_dummies(
                                                        sample_df['label']),
                                                    random_state=22)

# Instantiate Pipeline object: pl
pl = Pipeline([('clf', OneVsRestClassifier(LogisticRegression()))])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)


# Import the Imputer object

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']],
                                                    pd.get_dummies(
                                                        sample_df['label']),
                                                    random_state=456)

# Insantiate Pipeline object: pl
pl = Pipeline([
    ('imp', Imputer()),
    ('clf', OneVsRestClassifier(LogisticRegression()))
])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)


# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(
    lambda x: x[['numeric', 'with_missing']], validate=False)

# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())


# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(
                                                        sample_df['label']),
                                                    random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
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
)

# Instantiate nested pipeline: pl
pl = Pipeline([
    ('union', process_and_join_features),
    ('clf', OneVsRestClassifier(LogisticRegression()))
])


# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)
