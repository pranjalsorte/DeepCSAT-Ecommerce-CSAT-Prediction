"""
DeepCSAT Project
Step 6: Detailed Model Evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------------
# Load processed dataset
# -------------------------------------------------------

df = pd.read_csv("data/processed_data.csv")

# Separate features and target
X = df.drop("CSAT Score", axis=1)
y = df["CSAT Score"]

# Convert target to categorical
y_cat = to_categorical(y - 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for RNN/LSTM input
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# -------------------------------------------------------
# Load Trained Models
# -------------------------------------------------------

rnn_model = load_model("models/simple_rnn_model.h5")
lstm_model = load_model("models/lstm_model.h5")

# -------------------------------------------------------
# Predictions
# -------------------------------------------------------

rnn_pred = rnn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)

# Convert probabilities to class labels
rnn_pred_classes = np.argmax(rnn_pred, axis=1)
lstm_pred_classes = np.argmax(lstm_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

print("\n==============================")
print("SimpleRNN Classification Report")
print("==============================")

print(classification_report(y_true, rnn_pred_classes))

print("\nConfusion Matrix (SimpleRNN)")
print(confusion_matrix(y_true, rnn_pred_classes))

print("\n==============================")
print("LSTM Classification Report")
print("==============================")

print(classification_report(y_true, lstm_pred_classes))

print("\nConfusion Matrix (LSTM)")
print(confusion_matrix(y_true, lstm_pred_classes))