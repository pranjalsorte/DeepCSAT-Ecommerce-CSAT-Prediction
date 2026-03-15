"""
DeepCSAT Project
Model Training Script
"""

# ==============================
# Import Libraries
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input
from tensorflow.keras.utils import to_categorical

# ==============================
# Load Dataset
# ==============================

DATA_PATH = "data/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# ==============================
# Separate Features & Target
# ==============================

TARGET_COLUMN = "CSAT Score"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# ==============================
# Encode Categorical Columns
# ==============================

label_encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ==============================
# Scale Features
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# Encode Target Variable
# ==============================

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Convert to one-hot encoding
y_categorical = to_categorical(y_encoded)

# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==============================
# Reshape for RNN/LSTM
# (samples, timesteps, features)
# ==============================

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print("Training Shape:", X_train.shape)

# ==============================
# Handle Class Imbalance
# ==============================

y_labels = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_labels),
    y=y_labels
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ==============================
# Build SimpleRNN Model
# ==============================

simple_rnn = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    SimpleRNN(64, activation="tanh"),
    Dense(32, activation="relu"),
    Dense(y_train.shape[1], activation="softmax")
])

simple_rnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Train SimpleRNN
# ==============================

print("\nTraining SimpleRNN...")

simple_rnn.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

# ==============================
# Build LSTM Model
# ==============================

lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(y_train.shape[1], activation="softmax")
])

lstm_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Train LSTM
# ==============================

print("\nTraining LSTM...")

lstm_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

# ==============================
# Save Models
# ==============================

simple_rnn.save("models/simple_rnn_model.keras")
lstm_model.save("models/lstm_model.keras")

print("\n✅ Models trained and saved successfully!")