import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# load data
train_df = pd.read_csv("dataset/data_for_model/cleaned_train.csv")
test_df = pd.read_csv("dataset/data_for_model/cleaned_test.csv")






# vectorize data for train, validate and test
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train = vectorizer.fit_transform(train_df["answer"])
y_train = train_df["is_human"]

X_test = vectorizer.transform(test_df["answer"])
y_test = test_df["is_human"]

# use model to train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# test model
test_predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}")






# Prepare data
y_train = train_df["is_human"]
y_test = test_df["is_human"]

# Create pipeline (similar to before)
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words="english")),
#     ('clf', LogisticRegression(max_iter=1000))
# ])
#
# # Set parameters for grid search (same as before)
# parameters = {
#     'tfidf__max_features': [3000, 5000, 7000],
#     'clf__C': [0.1, 1, 10, 100],
#     'clf__penalty': ['l1', 'l2']
# }
#
# # Grid search using cross-validation
# grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
# grid_search.fit(train_df["answer"], y_train)
#
# # Best parameters and cross-validation score
# print("Best parameters:", grid_search.best_params_)
# print("Cross-validation accuracy:", grid_search.best_score_)
#
# # Evaluate on test set
# test_predictions = grid_search.predict(test_df["answer"])
# test_accuracy = accuracy_score(y_test, test_predictions)
# print(f"Test Accuracy: {test_accuracy:.2f}")

