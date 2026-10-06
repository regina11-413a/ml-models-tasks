from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="seara/rubert-tiny2-russian-sentiment"
)
texts = ["мне все зашло. все круто", "это было ужасно", "твердый середнячок"]

for text in texts:
    result = classifier(text)
    print(f"{text} → {result}")