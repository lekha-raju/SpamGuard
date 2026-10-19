import json, re, joblib
from flask import Flask, request, jsonify, render_template
from train import clean

app = Flask(__name__)
b = joblib.load("model.joblib"); vec, model = b["vec"], b["model"]
metrics = json.load(open("metrics.json")); vocab = vec.get_feature_names_out(); coef = model.coef_[0]

@app.route("/")
def home(): return render_template("index.html")

@app.route("/metrics")
def m(): return jsonify(metrics)

@app.route("/predict", methods=["POST"])
def predict():
    msg = (request.get_json().get("message") or "").strip()
    if len(msg) < 3: return jsonify(error="Type a message first"), 400
    X = vec.transform([clean(msg)]); p = float(model.predict_proba(X)[0][1])
    idx = X.nonzero()[1]
    kw = sorted(((vocab[i], coef[i] * X[0, i]) for i in idx), key=lambda t: abs(t[1]), reverse=True)[:8]
    flags = []
    if re.search(r"http\S+|www\.", msg, re.I): flags.append("Contains a link")
    if re.search(r"\b(free|win|won|prize|winner|claim|cash|urgent|offer)\b", msg, re.I): flags.append("Prize / urgency words")
    if re.search(r"[£$₹]\s?\d|\d+\s?(rs|lakh|crore)", msg, re.I): flags.append("Mentions money")
    if re.search(r"\d{5,}", msg): flags.append("Long number / phone number")
    if sum(c.isupper() for c in msg) / max(len(msg), 1) > .3: flags.append("Too many CAPITAL letters")
    return jsonify(verdict="SPAM" if p >= .5 else "HAM", confidence=round(max(p, 1 - p) * 100, 1),
                   flags=flags, keywords=[{"w": w, "spam": c > 0} for w, c in kw])

if __name__ == "__main__": app.run(debug=True)
