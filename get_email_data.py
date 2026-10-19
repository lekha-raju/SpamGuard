import io, re, csv, tarfile, urllib.request, email
from email import policy

BASE = "https://spamassassin.apache.org/old/publiccorpus/"
FILES = [("20030228_easy_ham.tar.bz2", "ham"), ("20030228_spam.tar.bz2", "spam"),
         ("20050311_spam_2.tar.bz2", "spam")]
rows = []
for name, label in FILES:
    try:
        print("Downloading", name, "...")
        data = urllib.request.urlopen(BASE + name, timeout=300).read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as t:
            for m in t.getmembers():
                if not m.isfile() or m.name.endswith("cmds"): continue
                msg = email.message_from_bytes(t.extractfile(m).read(), policy=policy.default)
                try:
                    body = msg.get_body(preferencelist=("plain", "html"))
                    text = body.get_content() if body else ""
                except Exception:
                    text = ""
                text = re.sub(r"<[^>]+>", " ", str(msg["subject"] or "") + " " + text)
                rows.append((label, re.sub(r"\s+", " ", text)[:3000]))
    except Exception as e:
        print("Skipped", name, e)

with open("email.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["label", "text"]); w.writerows(rows)
print("Done! email.csv created with", len(rows), "emails")
