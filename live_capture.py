# live_capture.py
import time
import pandas as pd
import json
import threading
from pynput import keyboard

# This function needs to be defined here or imported
def calculate_features(df):
    """Calculates keystroke dynamics features from a DataFrame."""
    if df.empty or len(df) < 2:
        return {}

    df['dwell_time'] = df['release_timestamp'] - df['press_timestamp']
    df = df.sort_values(by='press_timestamp')
    df['flight_time'] = df['press_timestamp'].shift(-1) - df['release_timestamp']

    dwell_times = df['dwell_time']
    flight_times = df['flight_time'].dropna()

    features = {
        'dwell_mean': dwell_times.mean(),
        'dwell_std': dwell_times.std(ddof=0),
        'dwell_median': dwell_times.median(),
        'dwell_min': dwell_times.min(),
        'dwell_max': dwell_times.max(),
        'flight_mean': flight_times.mean(),
        'flight_std': flight_times.std(ddof=0),
        'flight_median': flight_times.median(),
        'flight_min': flight_times.min(),
        'flight_max': flight_times.max(),
        'num_chars': len(df),
        'total_duration': df['release_timestamp'].max() - df['press_timestamp'].min()
    }
    features['typing_speed'] = features['num_chars'] / features['total_duration'] if features['total_duration'] > 0 else 0
    return features

# --- Global state for the listeners ---
prompt_key_events = []
is_prompt_capturing = False

background_key_events = []
background_listener_thread = None
background_listener_control = {'running': True}

# --- Section 1: Prompt-Based Capture (for initial login) ---
def on_prompt_press(key):
    if is_prompt_capturing:
        prompt_key_events.append({'type': 'press', 'key': key, 'timestamp': time.time()})

def on_prompt_release(key):
    global is_prompt_capturing
    if is_prompt_capturing:
        prompt_key_events.append({'type': 'release', 'key': key, 'timestamp': time.time()})
        if key == keyboard.Key.enter:
            is_prompt_capturing = False
            return False

def process_events(events):
    if len(events) < 5:
        print("Error: Not enough keystrokes captured.")
        return None
    presses, releases = {}, {}
    for event in events:
        try:
            key_repr = event['key'].char
        except AttributeError:
            key_repr = str(event['key'])
        
        timestamp_list = presses if event['type'] == 'press' else releases
        if key_repr not in timestamp_list:
            timestamp_list[key_repr] = []
        timestamp_list[key_repr].append(event['timestamp'])
    matched_events = []
    for key_repr, p_times in presses.items():
        if key_repr in releases:
            r_times = sorted(releases[key_repr])
            for p_time in sorted(p_times):
                matching_r_time = next((r for r in r_times if r > p_time), None)
                if matching_r_time:
                    matched_events.append({
                        'key': key_repr,
                        'press_timestamp': p_time,
                        'release_timestamp': matching_r_time
                    })
                    r_times.remove(matching_r_time)
    if not matched_events: return None
    df_live = pd.DataFrame(matched_events)
    features = calculate_features(df_live)
    if not features: return None
    try:
        with open('feature_order.json', 'r') as f:
            feature_order = json.load(f)
    except FileNotFoundError:
        feature_order = list(features.keys())
        with open('feature_order.json', 'w') as f:
            json.dump(feature_order, f)
    return [features.get(name, 0) for name in feature_order]

def capture_and_process_typing(prompt):
    global prompt_key_events, is_prompt_capturing
    prompt_key_events = []
    is_prompt_capturing = True
    print("\n[ Start typing the sentence below and press ENTER when finished ]")
    print("-----------------------------------------------------------------")
    print(prompt)
    print("-----------------------------------------------------------------")
    try:
        print("Get ready to type in 3...")
        time.sleep(1); print("2..."); time.sleep(1); print("1..."); time.sleep(1)
        print("Start typing now!")
    except KeyboardInterrupt:
        print("\nCapture cancelled."); return None
    prompt_key_events = []
    with keyboard.Listener(on_press=on_prompt_press, on_release=on_prompt_release) as listener:
        listener.join()
    print("\n-> Capture finished. Processing...")
    return process_events(prompt_key_events)

# --- Section 2: Background Capture (for continuous authentication) ---
def on_background_press(key):
    if background_listener_control['running']:
        background_key_events.append({'type': 'press', 'key': key, 'timestamp': time.time()})

def on_background_release(key):
    if background_listener_control['running']:
        background_key_events.append({'type': 'release', 'key': key, 'timestamp': time.time()})

def start_background_listener():
    global background_listener_thread
    if background_listener_thread is None:
        background_listener_control['running'] = True
        listener = keyboard.Listener(on_press=on_background_press, on_release=on_background_release)
        background_listener_thread = threading.Thread(target=listener.start, daemon=True)
        background_listener_thread.start()
        print("[SYSTEM] Continuous authentication monitor started.")

def stop_background_listener():
    global background_listener_thread
    if background_listener_thread is not None:
        background_listener_control['running'] = False
        background_listener_thread = None
        print("[SYSTEM] Continuous authentication monitor stopped.")

def get_and_clear_background_events():
    global background_key_events
    events = list(background_key_events)
    background_key_events = []
    return events
