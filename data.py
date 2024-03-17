import nltk
from nltk.corpus import movie_reviews

# Ensure the movie_reviews dataset is downloaded
nltk.download('movie_reviews')

# Load the movie_reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Now you can proceed with further preprocessing and analysis using the 'documents' variable
