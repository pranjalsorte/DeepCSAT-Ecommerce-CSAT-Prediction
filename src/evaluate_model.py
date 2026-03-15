"""
DeepCSAT Project
Step 5: Model Evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load processed data
df = pd.read_csv("data/processed_data.csv")

X = df.drop("CSAT Score", axis=1)
y = df["CSAT Score"]

# Convert target
y = to_categorical(y - 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Load models
rnn_model = load_model("models/simple_rnn_model.h5")
lstm_model = load_model("models/lstm_model.h5")

# Evaluate
print("\nEvaluating SimpleRNN...")
rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test)

print("\nEvaluating LSTM...")
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test)

print("\nResults")
print("SimpleRNN Accuracy:", rnn_acc)
print("LSTM Accuracy:", lstm_acc)