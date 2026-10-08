# authenticate_user.py
import json
import os
import sys
import numpy as np
from live_capture import capture_and_process_typing

PROFILES_DB = 'user_profiles.json'
CHALLENGE_PHRASSE = "the quick brown fox jumps over the lazy dog"
# This threshold has been lowered to a more realistic value.
# The new distance-based score is much stricter, so a lower
# threshold is needed to allow for natural typing variations.
SIMILARITY_THRESHOLD = 0.30 

def load_profiles():
    """Loads the user profiles database from the JSON file."""
    if not os.path.exists(PROFILES_DB):
        print("Error: User profiles database 'user_profiles.json' not found.")
        print("Please enroll at least one user using 'enroll_user.py' first.")
        sys.exit(1)
    with open(PROFILES_DB, 'r') as f:
        return json.load(f)

def calculate_similarity(vec_a, vec_b):
    """
    Calculates a similarity score based on normalized Euclidean distance.
    A score of 1.0 means a perfect match. The score decreases as the
    patterns become more different.
    """
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    # Calculate the Euclidean distance
    distance = np.linalg.norm(vec_a - vec_b)
    
    # Normalize the distance to a similarity score between 0 and 1
    # The formula 1 / (1 + distance) ensures that a distance of 0
    # results in a score of 1, and the score approaches 0 as distance increases.
    similarity = 1 / (1 + distance)
    
    return similarity

def main():
    """Main function to authenticate a user."""
    profiles = load_profiles()
    user_id = input("Enter your user ID to authenticate: ")

    if user_id not in profiles:
        print(f"Error: User '{user_id}' is not enrolled.")
        sys.exit(1)

    print(f"\n--- Authenticating User: {user_id} ---")
    print("Please type the following phrase to verify your identity.")

    # Capture live typing sample
    live_feature_vector = capture_and_process_typing(CHALLENGE_PHRASSE)

    if not live_feature_vector:
        print("\nAuthentication failed. Could not capture a valid typing sample.")
        sys.exit(1)

    # Retrieve the user's stored template
    stored_template = profiles[user_id]['template']

    # Compare the live sample to the stored template using the new method
    similarity_score = calculate_similarity(stored_template, live_feature_vector)

    print("\n--- Verification Result ---")
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Required Threshold: {SIMILARITY_THRESHOLD}")

    if similarity_score >= SIMILARITY_THRESHOLD:
        print("\n✅ SUCCESS: User identity verified.")
    else:
        print("\n❌ FAILED: Typing pattern does not match the enrolled template.")

if __name__ == "__main__":
    main()
