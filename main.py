from flask import Flask, render_template, request
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime

app = Flask(__name__)

# Define a list of weights to iterate through with more granularity
weights_list = [0.6, 0.7, 0.8, 0.9]

def calculate_cosine_similarity(doc1, doc2, weight):
    # Convert documents to vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([doc1, doc2]).toarray()

    # Apply weight only to the query vector
    weights_array = np.array([weight, 0.80 * weight])  # Apply weight to the query, and (1 - weight) to the title
    weighted_vectors = vectors * np.expand_dims(weights_array, axis=1)

    # Calculate cosine similarity
    similarity = cosine_similarity(weighted_vectors[0].reshape(1, -1), weighted_vectors[1].reshape(1, -1))
    return similarity[0][0]


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_precision(top_similarity_scores):
    relevant_results = [score for _, score, _ in top_similarity_scores]
    num_relevant = sum(0.9 for score in relevant_results if score > 0.5)  # Assuming a similarity score threshold of 0.5 for relevance
    return num_relevant / len(top_similarity_scores)


@app.route('/', methods=['GET'])
def search_view():
    return render_template('search.html', datetime=str(datetime.now()), weights_list=weights_list)


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    selected_weight = float(request.form['weight'])  # Get the selected weight as a float

    json_data = load_json_data('data.json')
    titles = json_data['titles']

    similarity_scores = []

    start_time = time.time()

    for title in titles:
        similarity_score = calculate_cosine_similarity(query, title, selected_weight)
        similarity_scores.append((title, similarity_score, selected_weight))

    end_time = time.time()
    processing_time = end_time - start_time

    # Sort the similarity scores by similarity score and then by title
    similarity_scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
    top_similarity_scores = similarity_scores[:5]

    precision = calculate_precision(top_similarity_scores)

    return render_template('result.html', query=query, similarity_scores=top_similarity_scores,
                           processing_time=processing_time, selected_weight=selected_weight, precision=precision)


if __name__ == '__main__':
    app.run()



# from flask import Flask, request, render_template
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# app = Flask(__name__)
#
#
# def calculate_cosine_similarity(doc1, doc2, weight):
#     # Convert documents to vectors using TF-IDF
#     vectorizer = TfidfVectorizer(stop_words='english')
#     vectors = vectorizer.fit_transform([doc1, doc2]).toarray()
#
#     # Apply weight only to the query vector
#     weights_array = np.array([weight, 0.8 * weight])  # Apply weight to the query, and (1 - weight) to the title
#     weighted_vectors = vectors * np.expand_dims(weights_array, axis=1)
#
#     # Calculate cosine similarity
#     similarity = cosine_similarity(weighted_vectors[0].reshape(1, -1), weighted_vectors[1].reshape(1, -1))
#     return similarity[0][0]
#
#
# @app.route('/')
# def index():
#     return render_template('search.html')
#
#
# @app.route('/calculate_similarity', methods=['POST'])
# def calculate_similarity():
#     doc1 = request.form['doc1']
#     doc2 = request.form['doc2']
#     weight = float(request.form['weight'])
#
#     similarity = calculate_cosine_similarity(doc1, doc2, weight)
#
#     return f'Cosine Similarity: {similarity}'
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

