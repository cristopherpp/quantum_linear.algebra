fom collections import defaultdict, Counter
import random

def load_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, text = line.split("\t", 1)
            data.append((int(label), text.lower().split()))
    return data

def text_to_features(words):
    return Counter(words)

def predict(words, weights, bias):
    features = text_to_features(words)
    score = bias
    for word, count in features.items():
        score += weights[word] * count
    return 1 if score >= 0 else 0, score

def train_perceptron(data, epochs=10, lr=0.1):
    weights = defaultdict(float)
    bias = 0.0

    for epoch in range(epochs):
        random.shuffle(data)
        mistakes = 0

        for label, words in data:
            pred, score = predict(words, weights, bias)
            error = label - pred

            if error != 0:
                mistakes += 1
                features = text_to_features(words)
                for word, count in features.items():
                    weights[word] += lr * error * count
                bias += lr * error

        print(f"epoch {epoch+1}: mistakes = {mistakes}")

    return weights, bias

def test_model(data, weights, bias):
    correct = 0
    for label, words in data:
        pred, score = predict(words, weights, bias)
        if pred == label:
            correct += 1
        print(f"text: {' '.join(words):30} label={label} pred={pred} score={score:.2f}")
    print(f"\naccuracy: {correct}/{len(data)} = {correct/len(data):.2%}")

def show_top_words(weights, top_n=10):
    items = sorted(weights.items(), key=lambda x: x[1])
    print("\nMost negative words:")
    for word, w in items[:top_n]:
        print(f"{word:15} {w:.3f}")

    print("\nMost positive words:")
    for word, w in items[-top_n:]:
        print(f"{word:15} {w:.3f}")

if __name__ == "__main__":
    data = load_data("dataset.txt")

    # simple split
    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data = data[split:]

    weights, bias = train_perceptron(train_data, epochs=15, lr=0.1)

    print("\n--- TEST ---")
    test_model(test_data, weights, bias)

    show_top_words(weights)
