import random
import nltk
from nltk.corpus import movie_reviews
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load the movie reviews dataset from NLTK
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(documents, columns=['review', 'sentiment'])

# Split the dataset into features and target variable
X = df['review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transform text data into numerical vectors using TF-IDF vectorizer
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize Linear Support Vector Classifier
svm_clf = LinearSVC()

# Train the classifier
svm_clf.fit(X_train_tfidf, y_train)

# Predict the sentiment on test data
y_pred = svm_clf.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
