from flask import Flask, render_template, request
import json
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords only (no punkt)
nltk.download("stopwords")

# Load FAQs
with open("faqs.json", "r") as f:
    faq_data = json.load(f)

faq_questions = [faq["question"] for faq in faq_data]
faq_answers = [faq["answer"] for faq in faq_data]

# Preprocess function (without punkt or word_tokenize)
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [word.strip(string.punctuation) for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

preprocessed_questions = [preprocess(q) for q in faq_questions]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        user_processed = preprocess(user_input)
        user_vec = vectorizer.transform([user_processed])

        similarity = cosine_similarity(user_vec, tfidf_matrix)
        best_match_index = similarity.argmax()

        if similarity[0][best_match_index] > 0.1:
            response = faq_answers[best_match_index]
        else:
            response = "Sorry, I don't understand that. Please try another question."

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
