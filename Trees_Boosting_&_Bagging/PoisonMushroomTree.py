# decision_tree_mushroom_with_test.py
# Build and test a decision tree classifier (no entropy, no ML libraries)

from collections import Counter, defaultdict

# ==========================
# Load Data
# ==========================
def load_data(path):
    """Load dataset from file (comma-separated values)."""
    with open(path, "r") as f:
        data = [line.strip().split(",") for line in f if line.strip()]
    return data

train_data = load_data("mush_train.data")
test_data = load_data("mush_test.data")

# ==========================
# Feature Names
# ==========================
features = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population",
    "habitat"
]

# ==========================
# Helper Functions
# ==========================
def majority_class(rows):
    """Return the most common class label."""
    labels = [row[0] for row in rows]
    return Counter(labels).most_common(1)[0][0]

def partition(rows, feature_index):
    """Split data based on feature values."""
    partitions = defaultdict(list)
    for row in rows:
        key = row[feature_index + 1]  # +1 because col 0 = class
        partitions[key].append(row)
    return partitions

def best_feature(rows, feature_indices):
    """Pick the feature that minimizes classification error."""
    best_feat = None
    lowest_error = float("inf")

    for i in feature_indices:
        parts = partition(rows, i)
        error = 0
        for subset in parts.values():
            labels = [r[0] for r in subset]
            majority = Counter(labels).most_common(1)[0][0]
            error += sum(1 for l in labels if l != majority)
        if error < lowest_error:
            lowest_error = error
            best_feat = i
    return best_feat

def build_tree(rows, feature_indices, depth=0):
    labels = [r[0] for r in rows]
    if len(set(labels)) == 1:
        return labels[0]

    if not feature_indices:
        return majority_class(rows)

    best_feat = best_feature(rows, feature_indices)
    if best_feat is None:
        return majority_class(rows)

    print(f"{'   ' * depth}Node depth {depth}: {features[best_feat]}")
    tree = {"feature": features[best_feat], "branches": {}}

    parts = partition(rows, best_feat)
    remaining = [i for i in feature_indices if i != best_feat]

    for val, subset in parts.items():
        print(f"{'  ' * depth}→ branch '{val}' ({len(subset)} samples)")
        tree["branches"][val] = build_tree(subset, remaining, depth + 1)

    return tree

# ==========================
# Prediction
# ==========================
def predict(tree, sample):
    while isinstance(tree, dict):
        feat = tree["feature"]
        idx = features.index(feat)
        val = sample[idx + 1]
        tree = tree["branches"][val]
    return tree


def test_accuracy(tree, test_data):

    correct = 0
    total = len(test_data)
    confusion = Counter()

    for row in test_data:
        actual = row[0]
        pred = predict(tree, row)
        if pred == actual:
            correct += 1
        confusion[(actual, pred)] += 1

    accuracy = correct / total * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nConfusion Summary (actual, predicted):")
    for (actual, pred), count in confusion.items():
        print(f"  {actual} → {pred}: {count} samples")


feature_indices = list(range(len(features)))
print("Training decision tree...\n")

tree = build_tree(train_data, feature_indices)
print("\nTree construction complete!")

print("\nEvaluating on test data...")
test_accuracy(tree, test_data)
