import io, urllib.request, zipfile

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
print("Downloading dataset...")
data = urllib.request.urlopen(URL, timeout=60).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    raw = z.read("SMSSpamCollection")
open("spam.csv", "wb").write(raw)
print("Done! spam.csv created.")
