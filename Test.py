# Import the code from sentiment_analysis.py
from model import *

# Now you can use X_test and svm_clf in this file
# Example:
# y_pred = svm_clf.predict(X_test)
# print(y_pred)

# Test with sample reviews
sample_reviews = [
    "This movie was fantastic! I loved every moment of it.",
    "The acting was terrible, and the plot was boring.",
    "I was pleasantly surprised by how good this movie was.",
    "I couldn't stand this film. It was a waste of time."
]

# Transform the sample reviews into numerical vectors using the TF-IDF vectorizer
sample_reviews_tfidf = tfidf_vectorizer.transform(sample_reviews)

# Predict the sentiment of the sample reviews using the trained model
predicted_sentiments = svm_clf.predict(sample_reviews_tfidf)

# Print the predicted sentiments for each review
for review, sentiment in zip(sample_reviews, predicted_sentiments):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {'Positive' if sentiment == 'pos' else 'Negative'}")
    print()