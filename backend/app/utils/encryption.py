import os
import logging
from cryptography.fernet import Fernet
from flask import current_app

logger = logging.getLogger(__name__)

_fernet_instance = None
_cached_key_bytes = None

def get_active_key_bytes():
    """Retrieve the primary FIELD_ENCRYPTION_KEY bytes from app config or environment."""
    key = None
    if current_app:
        key = current_app.config.get("FIELD_ENCRYPTION_KEY")
    if not key:
        key = os.environ.get("FIELD_ENCRYPTION_KEY")
    if not key:
        return None
    if isinstance(key, str):
        return key.strip().encode("utf-8")
    return key

def get_fernet():
    """
    Returns the primary Fernet cipher instance using the configured persistent key.
    Fails safely without hardcoded fallbacks or ephemeral key generation when no key is set.
    """
    global _fernet_instance, _cached_key_bytes
    key_bytes = get_active_key_bytes()
    if not key_bytes:
        is_prod = (
            (current_app and current_app.config.get("FLASK_ENV") == "production")
            or os.environ.get("FLASK_ENV") == "production"
        )
        if is_prod:
            logger.error("FIELD_ENCRYPTION_KEY is missing in production environment.")
            raise RuntimeError("CRITICAL SECURITY ERROR: FIELD_ENCRYPTION_KEY is required and must be configured in production.")
        raise RuntimeError("FIELD_ENCRYPTION_KEY is required but not configured.")

    if _fernet_instance is not None and _cached_key_bytes == key_bytes:
        return _fernet_instance

    try:
        instance = Fernet(key_bytes)
        _fernet_instance = instance
        _cached_key_bytes = key_bytes
        return instance
    except Exception as e:
        logger.error(f"Configured FIELD_ENCRYPTION_KEY is invalid for Fernet cipher: {type(e).__name__}")
        raise RuntimeError("Configured FIELD_ENCRYPTION_KEY is invalid for Fernet cipher.")

def get_fallback_ciphers():
    """
    Retrieve configured historical/legacy Fernet ciphers for backward compatibility.
    Reads from FIELD_ENCRYPTION_FALLBACK_KEYS or PREVIOUS_FIELD_ENCRYPTION_KEYS configuration.
    """
    raw_fallbacks = None
    if current_app:
        raw_fallbacks = current_app.config.get("FIELD_ENCRYPTION_FALLBACK_KEYS") or current_app.config.get("PREVIOUS_FIELD_ENCRYPTION_KEYS")
    if not raw_fallbacks:
        raw_fallbacks = os.environ.get("FIELD_ENCRYPTION_FALLBACK_KEYS") or os.environ.get("PREVIOUS_FIELD_ENCRYPTION_KEYS")

    if not raw_fallbacks:
        return []

    if isinstance(raw_fallbacks, str):
        key_list = [k.strip() for k in raw_fallbacks.split(",") if k.strip()]
    elif isinstance(raw_fallbacks, (list, tuple)):
        key_list = raw_fallbacks
    else:
        key_list = []

    ciphers = []
    for k in key_list:
        k_bytes = k.encode("utf-8") if isinstance(k, str) else k
        try:
            ciphers.append(Fernet(k_bytes))
        except Exception:
            logger.warning("Configured fallback encryption key could not be parsed as a valid Fernet key.")
    return ciphers

def encrypt_text(plain_text: str) -> bytes:
    """Encrypt raw document text using configured Fernet/AES before storing at rest."""
    if not plain_text:
        return b""
    f = get_fernet()
    return f.encrypt(plain_text.encode("utf-8"))

def decrypt_text(encrypted_bytes: bytes) -> str:
    """
    Decrypt encrypted document text with multi-key recovery for configured legacy keys.
    Tries the active key first; falls back to configured fallback keys if needed.
    """
    if not encrypted_bytes:
        return ""
    f = get_fernet()
    try:
        return f.decrypt(encrypted_bytes).decode("utf-8")
    except Exception as primary_err:
        for fallback_cipher in get_fallback_ciphers():
            try:
                decrypted = fallback_cipher.decrypt(encrypted_bytes).decode("utf-8")
                logger.info("Successfully decrypted ciphertext using a configured fallback encryption key.")
                return decrypted
            except Exception:
                continue
        logger.error(f"Failed to decrypt document text: {type(primary_err).__name__}")
        raise primary_err
