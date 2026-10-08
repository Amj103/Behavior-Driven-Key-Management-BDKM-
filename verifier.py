# verifier.py
import joblib
import json
import numpy as np
import hashlib
import os

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL = joblib.load(os.path.join(BASE_DIR, 'behavior_rf.pkl'))
    SCALER = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    with open(os.path.join(BASE_DIR, 'label_map.json'), 'r') as f:
        LABEL_MAP = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading verification artifacts: {e}")
    MODEL, SCALER, LABEL_MAP = None, None, None

def verify(claimed_user: str, raw_feature_vector: list, threshold: float = 0.5) -> tuple:
    """
    Verifies a user's identity based on their typing dynamics.

    Returns:
        A tuple containing:
        - bool: True if authentication is successful, False otherwise.
        - float: The model's confidence score for the claimed user.
        - np.array: The scaled feature vector.
    """
    if not all([MODEL, SCALER, LABEL_MAP]):
        raise RuntimeError("Verification module not initialized.")

    if str(claimed_user) not in LABEL_MAP:
        return False, 0.0, None

    claimed_user_int_label = LABEL_MAP[str(claimed_user)]
    feature_vector_np = np.array(raw_feature_vector).reshape(1, -1)
    scaled_vector = SCALER.transform(feature_vector_np)
    probabilities = MODEL.predict_proba(scaled_vector)[0]
    score = probabilities[claimed_user_int_label]
    accepted = score >= threshold

    return accepted, score, scaled_vector.flatten()

def derive_key_from_scaled_vector(scaled_vector: np.array) -> bytes:
    """Derives a deterministic 32-byte AES key from a feature vector."""
    vector_bytes = scaled_vector.tobytes()
    return hashlib.sha256(vector_bytes).digest()
