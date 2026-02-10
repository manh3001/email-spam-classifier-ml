import joblib

# load model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_spam(text):
    text_vector = vectorizer.transform([text])
    result = model.predict(text_vector)[0]
    return "SPAM" if result == 1 else "HAM"

# test
msg = input("Enter a message: ")
print("Prediction:", predict_spam(msg))
