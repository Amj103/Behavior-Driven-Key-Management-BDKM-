# preprocess.py
import pandas as pd
import numpy as np
import sys
import os

def calculate_features(df_sample):
    """
    Calculates typing features for a single sample of keystroke data.

    Args:
        df_sample (pd.DataFrame): A DataFrame containing the keystroke data
                                  for one sample (one participant, one session).
                                  It must have 'press_timestamp' and
                                  'release_timestamp' columns.

    Returns:
        dict: A dictionary containing all the calculated features for the sample.
              Returns None if the sample is empty or invalid.
    """
    if df_sample.empty or len(df_sample) < 2:
        return None

    # Ensure data is sorted by press time to correctly calculate flight times
    df_sample = df_sample.sort_values('press_timestamp').reset_index(drop=True)

    # --- Feature Calculation ---

    # 1. Dwell Times: The duration each key is held down.
    dwell_times = df_sample['release_timestamp'] - df_sample['press_timestamp']
    dwell_times = dwell_times[dwell_times > 0]
    if dwell_times.empty:
        return None

    # 2. Flight Times: The time between releasing one key and pressing the next.
    flight_times = df_sample['press_timestamp'].iloc[1:].values - df_sample['release_timestamp'].iloc[:-1].values
    flight_times = flight_times[flight_times > 0]
    if len(flight_times) == 0:
        flight_times = np.array([0])

    # 3. Total Duration, Chars, and Typing Speed
    total_duration = df_sample['release_timestamp'].max() - df_sample['press_timestamp'].min()
    num_chars = len(df_sample)
    typing_speed = num_chars / total_duration if total_duration > 0 else 0

    # Assemble features into a dictionary
    features = {
        'dwell_mean': dwell_times.mean(),
        'dwell_std': dwell_times.std(ddof=0),
        'dwell_median': dwell_times.median(),
        'dwell_min': dwell_times.min(),
        'dwell_max': dwell_times.max(),
        'flight_mean': np.mean(flight_times),
        'flight_std': np.std(flight_times, ddof=0),
        'flight_median': np.median(flight_times),
        'flight_min': np.min(flight_times),
        'flight_max': np.max(flight_times),
        'num_chars': num_chars,
        'typing_speed': typing_speed,
        'total_duration': total_duration
    }

    return features

def main(input_csv_path, output_csv_path):
    """Main function to drive the preprocessing of the keystroke data."""
    print(f"-> Loading raw keystroke data from '{input_csv_path}'...")
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at '{input_csv_path}'.")
        sys.exit(1)

    col_names = ['participant_id', 'sample_id', 'key', 'press_timestamp', 'release_timestamp']
    
    # --- FIX ---
    # Use `usecols=range(5)` to read only the first 5 columns. This prevents
    # DtypeWarning and parsing errors if the CSV has extra, malformed columns.
    df = pd.read_csv(input_csv_path, header=None, names=col_names, usecols=range(5))

    df['press_timestamp'] = pd.to_numeric(df['press_timestamp'], errors='coerce')
    df['release_timestamp'] = pd.to_numeric(df['release_timestamp'], errors='coerce')
    df.dropna(inplace=True)

    print("-> Grouping data and calculating features...")
    grouped = df.groupby(['participant_id', 'sample_id'])
    
    feature_list = []
    for (participant, sample), group_df in grouped:
        features = calculate_features(group_df)
        if features:
            features['participant_id'] = participant
            features['sample_id'] = sample
            feature_list.append(features)

    if not feature_list:
        print("Error: No valid features could be extracted. Please check the format of 'raw_keystrokes.csv'.")
        sys.exit(1)

    features_df = pd.DataFrame(feature_list)
    cols = ['participant_id', 'sample_id'] + [col for col in features_df.columns if col not in ['participant_id', 'sample_id']]
    features_df = features_df[cols]

    print(f"-> Saving processed features to '{output_csv_path}'...")
    features_df.to_csv(output_csv_path, index=False)
    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <path_to_raw_data.csv> <path_to_output_features.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
