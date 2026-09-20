import joblib
import json
import preprocess
import db
from app import predict_sentiment_single, model, vectorizer

# 1. Check model instances
print("[1] Model type:", type(model))
print("[2] Vectorizer type:", type(vectorizer))
assert model is not None, "Model should not be None"
assert vectorizer is not None, "Vectorizer should not be None"

# 2. Check direct inference vs app inference
test_text = "The doctor was prompt, kind, and gave fantastic medical advice."
app_res = predict_sentiment_single(test_text)

cleaned = preprocess.preprocess_comment(test_text)
x_vec = vectorizer.transform([cleaned])
probs = model.predict_proba(x_vec)[0]
classes = list(model.classes_)
pos_idx = classes.index("positive")
pos_prob = float(probs[pos_idx])

print("[3] App result sentiment:", app_res["sentiment"])
print("[4] App result confidence:", app_res["confidence"])
print("[5] Direct model positive prob:", round(pos_prob, 4))
assert app_res["sentiment"] == "positive"
assert abs(app_res["confidence"] - round(pos_prob, 4)) < 1e-4

print("ITEM 5 VERIFIED: Prediction directly uses the saved Scikit-learn model and TF-IDF vectorizer!")
