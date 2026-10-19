"""Run: python train.py   (needs spam.csv in this folder)"""
import json, re, joblib, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def clean(t):
    t = re.sub(r"http\S+|www\S+", " urltoken ", str(t).lower())
    t = re.sub(r"\d{5,}", " numtoken ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", t)).strip()

def load():
    try:   # Kaggle format: v1 (label), v2 (text)
        d = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    except Exception:   # UCI format: tab separated, no header
        d = pd.read_csv("spam.csv", sep="\t", header=None, encoding="latin-1")
    d.columns = ["label", "text"]; return d

if __name__ == "__main__":
    df = load().drop_duplicates()
    df["y"] = (df["label"] == "spam").astype(int); df["c"] = df["text"].map(clean)
    Xtr, Xte, ytr, yte = train_test_split(df["c"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    models = {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
              "Linear SVC": LinearSVC(class_weight="balanced"), "Random Forest": RandomForestClassifier(200, n_jobs=-1, random_state=42)}
    res, fitted = {}, {}
    for n, m in models.items():
        m.fit(A, ytr); p = m.predict(B); fitted[n] = (m, p)
        res[n] = {k: round(f(yte, p) * 100, 2) for k, f in
                  [("accuracy", accuracy_score), ("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]}
        print(n, res[n])
    best = max(res, key=lambda n: res[n]["f1"])
    # prediction model: Logistic Regression (gives probability + explainable words)
    joblib.dump({"vec": vec, "model": models["Logistic Regression"]}, "model.joblib")
    json.dump({"best": best, "results": res, "train": len(Xtr), "test": len(Xte)}, open("metrics.json", "w"))
    # charts
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = list(res); ax[0].bar(names, [res[n]["f1"] for n in names], color="#6c7bff")
    ax[0].set_title("F1-score by model"); ax[0].set_ylim(80, 100); ax[0].tick_params(axis="x", rotation=20)
    cm = confusion_matrix(yte, fitted[best][1]); ax[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax[1].set_xticks([0, 1], ["Ham", "Spam"]); ax[1].set_yticks([0, 1], ["Ham", "Spam"])
    ax[1].set_title(f"Confusion Matrix ({best})"); plt.tight_layout(); plt.savefig("static/charts.png", dpi=120)
    print("Best:", best, "| Done!")
