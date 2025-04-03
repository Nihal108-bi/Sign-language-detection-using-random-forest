import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Determine max length of feature vectors
max_length = max(len(sample) for sample in data_dict['data'])

# Pad or truncate feature vectors
def pad_or_truncate(data, max_length):
    return [sample[:max_length] + [0] * (max_length - len(sample)) for sample in data]

data = np.array(pad_or_truncate(data_dict['data'], max_length))
labels = np.array(data_dict['labels'])

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate Model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save Model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model successfully saved as model.p")
