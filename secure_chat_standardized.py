# secure_chat_standardized.py
import socket
import threading
import sys
import time
import os
import json
import numpy as np
import hashlib
import platform # Added for the fix

# Platform-specific imports for clearing the input buffer
if platform.system() == "Windows":
    import msvcrt
else:
    # For Linux/macOS
    import termios
    import tty

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# --- FINGERPRINT ---
print("--- RUNNING VERSION 3.1: BUFFER FIX ---")
# -------------------

# Import modules from our new enrollment/authentication system
from live_capture import capture_and_process_typing, start_background_listener, stop_background_listener, get_and_clear_background_events, process_events
from enroll_user import load_profiles, save_profiles

# --- Constants ---
REKEY_INTERVAL_SECONDS = 900  # 15 minutes
REKEY_MESSAGE_COUNT = 20 # Lowered for easier testing
AUTH_THRESHOLD = 0.60 # Adjusted for the new similarity metric
CHALLENGE_PHRASE = "the quick brown fox jumps over the lazy dog"
PROFILES_DB = 'user_profiles.json'

# --- FIX: New helper function to clear the input buffer ---
def clear_input_buffer():
    """Clears any characters waiting in the standard input buffer."""
    try:
        if platform.system() == "Windows":
            while msvcrt.kbhit():
                msvcrt.getch()
        else:
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except Exception:
        # Fails in some environments (e.g., IDEs), but that's okay.
        # The primary goal is to fix it for the user's command prompt.
        pass

# --- Cryptography Helpers ---
def generate_rsa_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    return private_key, private_key.public_key()

def serialize_public_key(pub_key):
    return pub_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)

def deserialize_public_key(pem_data):
    return serialization.load_pem_public_key(pem_data, backend=default_backend())

def rsa_encrypt(pub_key, data):
    return pub_key.encrypt(data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))

def rsa_decrypt(priv_key, encrypted_data):
    return priv_key.decrypt(encrypted_data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))

def aes_encrypt(key, iv, plaintext):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    pad_len = 16 - (len(plaintext) % 16)
    padded_data = plaintext.encode() + bytes([pad_len] * pad_len)
    return encryptor.update(padded_data) + encryptor.finalize()

def aes_decrypt(key, iv, ciphertext):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    pad_len = padded_plaintext[-1]
    return padded_plaintext[:-pad_len].decode()

def derive_key_from_feature_vector(feature_vector: list) -> bytes:
    vector_bytes = np.array(feature_vector).tobytes()
    return hashlib.sha256(vector_bytes).digest()

# --- Server Logic ---
def receive_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError(f"Socket closed before receiving all {length} bytes.")
        data += more
    return data

def handle_client(client_socket, rsa_private_key):
    client_address = client_socket.getpeername()
    print(f"[SERVER] New connection from {client_address}")
    try:
        public_key_pem = serialize_public_key(rsa_private_key.public_key())
        client_socket.sendall(public_key_pem)
        
        encrypted_session_key = receive_all(client_socket, 256)
        session_key_data = rsa_decrypt(rsa_private_key, encrypted_session_key)
        aes_key = session_key_data[:32]
        iv = session_key_data[32:]
        print(f"[SERVER] Secure channel established with {client_address}")
        client_socket.sendall(b"ACK_KEY")

        while True:
            encrypted_msg = client_socket.recv(1024)
            if not encrypted_msg: break
            
            message = aes_decrypt(aes_key, iv, encrypted_msg)
            
            if message == "REKEY_INITIATE":
                print(f"[SERVER] Client {client_address} initiated re-keying.")
                client_socket.sendall(b"REKEY_ACK")
                new_encrypted_session_key = receive_all(client_socket, 256)
                new_session_key_data = rsa_decrypt(rsa_private_key, new_encrypted_session_key)
                aes_key = new_session_key_data[:32]
                iv = new_session_key_data[32:]
                print(f"[SERVER] Re-keying successful with {client_address}.")
                client_socket.sendall(b"REKEY_COMPLETE")
                continue

            print(f"[CLIENT {client_address}] SAYS: {message}")
            
    except Exception as e:
        print(f"[SERVER] Error with {client_address}: {e}")
    finally:
        print(f"[SERVER] Client {client_address} disconnected.")
        client_socket.close()

def run_server(host, port):
    private_key, _ = generate_rsa_keys()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[SERVER] Listening on {host}:{port}")
    while True:
        client_socket, _ = server_socket.accept()
        threading.Thread(target=handle_client, args=(client_socket, private_key), daemon=True).start()

# --- Client Logic ---
class ChatClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.user_id = ""
        self.server_public_key = None
        self.aes_key = None
        self.iv = None
        self.message_count = 0
        self.last_rekey_time = time.time()
        self.rekey_lock = threading.Lock()
        self.is_rekeying = False

    def calculate_similarity_standardized(self, stored_profile, live_features):
        stored_template = np.array(stored_profile['template'])
        stored_mean = stored_profile['mean']
        stored_std = stored_profile['std']
        
        live_features = np.array(live_features)

        if stored_std == 0: stored_std = 1.0 # Avoid division by zero
        standardized_live = (live_features - stored_mean) / stored_std
        standardized_template = (stored_template - stored_mean) / stored_std

        distance = np.linalg.norm(standardized_template - standardized_live)
        similarity = 1 / (1 + distance)
        return similarity

    def perform_authentication(self, is_initial=True):
        profiles = load_profiles()
        if self.user_id not in profiles:
            print(f"Error: User '{self.user_id}' is not enrolled.")
            return None, 0.0

        if is_initial:
            print(f"\n--- Authenticating User: {self.user_id} ---")
            live_features = capture_and_process_typing(CHALLENGE_PHRASE)
        else:
            events = get_and_clear_background_events()
            live_features = process_events(events)

        if not live_features:
            print("Error: Not enough keystrokes captured.")
            print("Authentication failed: Could not capture a valid typing sample.")
            return None

        stored_profile = profiles[self.user_id]
        similarity = self.calculate_similarity_standardized(stored_profile, live_features)

        print(f"\n--- Verification Result ---")
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Required Threshold: {AUTH_THRESHOLD}")

        if similarity >= AUTH_THRESHOLD:
            print("\n✅ SUCCESS: User identity verified.")
            return live_features
        else:
            print("\n❌ FAILED: Typing pattern does not match.")
            return None

    def perform_enrollment(self):
        profiles = load_profiles()
        if self.user_id in profiles:
            print(f"User '{self.user_id}' already exists.")
            return False
        
        print(f"\n--- Enrolling User: {self.user_id} ---")
        feature_vector = capture_and_process_typing(CHALLENGE_PHRASE)
        if not feature_vector:
            print("Enrollment failed.")
            return False
        
        template_np = np.array(feature_vector)
        mean = np.mean(template_np)
        std = np.std(template_np)
        if std == 0: std = 1.0

        profiles[self.user_id] = {
            "template": feature_vector,
            "mean": mean,
            "std": std
        }
        save_profiles(profiles)
        print(f"\n✅ User '{self.user_id}' enrolled successfully!")
        return True

    def initiate_rekey_seamless(self):
        with self.rekey_lock:
            self.is_rekeying = True
            print("\n[SYSTEM] Seamlessly verifying identity for key rotation...")
            
            try:
                self.sock.settimeout(None)
                
                feature_vector = self.perform_authentication(is_initial=False)

                if not feature_vector:
                    print("[SYSTEM] ❌ Verification failed. Continuing with old key.")
                    self.last_rekey_time = time.time()
                    self.message_count = 0 # Also reset message count
                    return

                print("[SYSTEM] ✅ Identity confirmed.")
                new_aes_key = derive_key_from_feature_vector(feature_vector)
                new_iv = os.urandom(16)

                self.sock.sendall(aes_encrypt(self.aes_key, self.iv, "REKEY_INITIATE"))
                if self.sock.recv(1024) != b"REKEY_ACK": return

                session_data = new_aes_key + new_iv
                encrypted_data = rsa_encrypt(self.server_public_key, session_data)
                self.sock.sendall(encrypted_data)
                
                if self.sock.recv(1024) == b"REKEY_COMPLETE":
                    self.aes_key, self.iv = new_aes_key, new_iv
                    self.message_count, self.last_rekey_time = 0, time.time()
                    print("[SYSTEM] ✅ Key rotation successful.")
                else:
                    print("[SYSTEM] Re-key finalization failed.")
            except Exception as e:
                print(f"[SYSTEM] Error during re-key: {e}")
            finally:
                self.sock.settimeout(1.0)
                self.is_rekeying = False

    def receive_messages(self):
        while True:
            try:
                if not self.is_rekeying:
                    try:
                        encrypted_response = self.sock.recv(1024)
                        if not encrypted_response: break
                        response = aes_decrypt(self.aes_key, self.iv, encrypted_response)
                        # Clear line and print
                        sys.stdout.write('\r' + ' ' * 80 + '\r') 
                        print(f"[SERVER SAYS] {response}")
                        print("[YOU] > ", end="", flush=True)
                    except socket.timeout:
                        continue
                else:
                    time.sleep(0.2)
            except Exception:
                break
        print("\n[CLIENT] Server connection lost.")
        os._exit(0)

    def start(self):
        try:
            self.sock.connect((self.host, self.port))
            print(f"[CLIENT] Connected to server.")
            public_key_pem = self.sock.recv(1024)
            self.server_public_key = deserialize_public_key(public_key_pem)

            feature_vector = None
            while not feature_vector:
                action = input("Do you want to [enroll] or [login]? ").lower()
                self.user_id = input(f"Enter user ID to {action}: ")

                if action == 'enroll':
                    self.perform_enrollment()
                    print("\nPlease login with your new credentials.")
                    continue
                elif action == 'login':
                    feature_vector = self.perform_authentication(is_initial=True)
                    if not feature_vector:
                        print("Login failed. Please try again.")
                else:
                    print("Invalid action.")

            self.aes_key = derive_key_from_feature_vector(feature_vector)
            self.iv = os.urandom(16)
            session_data = self.aes_key + self.iv
            encrypted_data = rsa_encrypt(self.server_public_key, session_data)
            self.sock.sendall(encrypted_data)
            if self.sock.recv(1024) != b"ACK_KEY": return
            
            print("\n[CLIENT] Secure channel established.")
            self.sock.settimeout(1.0)
            start_background_listener()
            threading.Thread(target=self.receive_messages, daemon=True).start()
            self.last_rekey_time = time.time()

            # --- THE FIX IS APPLIED HERE ---
            clear_input_buffer()
            # --------------------------------

            print("[YOU] > ", end="", flush=True)
            while True:
                message = sys.stdin.readline().strip()
                if message.lower() == 'exit': break
                
                with self.rekey_lock:
                    if not self.is_rekeying:
                        self.sock.sendall(aes_encrypt(self.aes_key, self.iv, message))
                        self.message_count += 1

                if time.time() - self.last_rekey_time > REKEY_INTERVAL_SECONDS or \
                   self.message_count >= REKEY_MESSAGE_COUNT:
                    self.initiate_rekey_seamless()
                    print("[YOU] > ", end="", flush=True)
        except Exception as e:
            print(f"[CLIENT] An error occurred: {e}")
        finally:
            stop_background_listener()
            self.sock.close()
            print("[CLIENT] Connection closed.")
            os._exit(0)

def main():
    if len(sys.argv) < 4:
        print("Usage:\n  Server: python secure_chat_standardized.py server <host> <port>\n  Client: python secure_chat_standardized.py client <host> <port>")
        sys.exit(1)
    mode, host, port = sys.argv[1].lower(), sys.argv[2], int(sys.argv[3])
    if mode == 'server':
        run_server(host, port)
    elif mode == 'client':
        ChatClient(host, port).start()
    else:
        print(f"Invalid mode '{mode}'.")

if __name__ == "__main__":
    main()

