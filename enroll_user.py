# enroll_user.py
import json
import os
import numpy as np
from live_capture import capture_and_process_typing

PROFILES_DB = 'user_profiles.json'
CHALLENGE_PHRASE = "the quick brown fox jumps over the lazy dog"

def load_profiles():
    """Loads user profiles from the JSON database."""
    if os.path.exists(PROFILES_DB):
        try:
            with open(PROFILES_DB, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {} # Return empty dict if file is corrupted or empty
    return {}

def save_profiles(profiles):
    """Saves the user profiles to the JSON database."""
    with open(PROFILES_DB, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(o):
            if isinstance(o, np.generic): return o.item()  
            raise TypeError
        json.dump(profiles, f, indent=4, default=convert)

def main():
    """Main function to enroll a new user."""
    profiles = load_profiles()
    
    user_id = input("Enter a new user ID to enroll: ")
    if user_id in profiles:
        print(f"Error: User '{user_id}' already exists.")
        return

    print(f"\n--- Enrolling User: {user_id} ---")
    
    feature_vector = capture_and_process_typing(CHALLENGE_PHRASE)
    
    if feature_vector:
        template_np = np.array(feature_vector)
        mean = np.mean(template_np)
        std = np.std(template_np)
        if std == 0: std = 1.0

        profiles[user_id] = {
            "template": feature_vector,
            "mean": mean,
            "std": std
        }
        save_profiles(profiles)
        print(f"\n✅ User '{user_id}' enrolled successfully!")
    else:
        print("\nEnrollment failed. Could not capture a valid typing sample.")

if __name__ == "__main__":
    main()
