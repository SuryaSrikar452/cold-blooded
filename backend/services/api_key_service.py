import hashlib
import secrets
import time
from datetime import datetime

def generate_raw_api_key():
    """Generates a secure random API key prefixed with 'sh405_'."""
    return f"sh405_live_{secrets.token_hex(16)}"

def hash_api_key(raw_key):
    """Calculates SHA-256 hash of an API key for safe database storage."""
    return hashlib.sha256(raw_key.strip().encode('utf-8')).hexdigest()

def verify_api_key(raw_key, stored_hash):
    """Verifies a raw API key against a stored SHA-256 hash."""
    return hash_api_key(raw_key) == stored_hash
