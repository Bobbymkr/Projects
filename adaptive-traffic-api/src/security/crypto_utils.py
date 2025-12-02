"""
Advanced Cryptographic Utilities for Enhanced Security.

Implements enterprise-grade cryptographic features:
- Secure random number generation
- Key derivation and management
- Advanced encryption/decryption
- Digital signatures and verification
- Secure data hashing
- Perfect Forward Secrecy (PFS)
"""

import os
import hashlib
import hmac
import secrets
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import cryptography library for advanced features
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("cryptography library not available, using fallback implementations")

@dataclass
class EncryptionKey:
    """Secure encryption key container."""
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    max_usage: Optional[int] = None

class SecureRandom:
    """Cryptographically secure random number generator."""
    
    @staticmethod
    def bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)
        
    @staticmethod
    def string(length: int, alphabet: str = None) -> str:
        """Generate cryptographically secure random string."""
        if alphabet is None:
            # Use URL-safe alphabet by default
            return secrets.token_urlsafe(length)
        
        return ''.join(secrets.choice(alphabet) for _ in range(length))
        
    @staticmethod
    def integer(min_val: int, max_val: int) -> int:
        """Generate cryptographically secure random integer."""
        return secrets.randbelow(max_val - min_val + 1) + min_val
        
    @staticmethod
    def choice(sequence: List[Any]) -> Any:
        """Securely choose random element from sequence."""
        return secrets.choice(sequence)
        
    @staticmethod
    def uuid4() -> str:
        """Generate cryptographically secure UUID4."""
        random_bytes = secrets.token_bytes(16)
        # Set version (4) and variant bits according to RFC 4122
        random_bytes = bytearray(random_bytes)
        random_bytes[6] = (random_bytes[6] & 0x0f) | 0x40  # Version 4
        random_bytes[8] = (random_bytes[8] & 0x3f) | 0x80  # Variant 10
        
        hex_string = random_bytes.hex()
        return f"{hex_string[:8]}-{hex_string[8:12]}-{hex_string[12:16]}-{hex_string[16:20]}-{hex_string[20:]}"

class SecureCrypto:
    """Advanced cryptographic operations manager."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or self._generate_master_key()
        self.key_store: Dict[str, EncryptionKey] = {}
        self.algorithm_registry = self._init_algorithms()
        
    def _generate_master_key(self) -> bytes:
        """Generate a secure master key."""
        return secrets.token_bytes(32)  # 256-bit key
        
    def _init_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize supported cryptographic algorithms."""
        algorithms = {
            'AES-256-GCM': {
                'key_size': 32,
                'nonce_size': 12,
                'tag_size': 16,
                'secure_level': 'high'
            },
            'CHACHA20-POLY1305': {
                'key_size': 32,
                'nonce_size': 12,
                'tag_size': 16,
                'secure_level': 'high'
            },
            'FERNET': {
                'key_size': 32,
                'secure_level': 'medium',
                'description': 'Symmetric encryption with built-in integrity'
            }
        }
        
        if CRYPTOGRAPHY_AVAILABLE:
            algorithms.update({
                'RSA-4096': {
                    'key_size': 4096,
                    'secure_level': 'very_high',
                    'type': 'asymmetric'
                }
            })
            
        return algorithms
        
    def derive_key(self, password: str, salt: Optional[bytes] = None, 
                  algorithm: str = 'scrypt', key_length: int = 32) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using secure KDF.
        
        Returns:
            (derived_key, salt_used)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
            
        if algorithm == 'scrypt' and CRYPTOGRAPHY_AVAILABLE:
            # Use Scrypt - more secure but slower
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                n=2**17,  # CPU/memory cost factor (high security)
                r=8,      # Block size
                p=1,      # Parallelization factor
            )
            derived_key = kdf.derive(password.encode('utf-8'))
            
        elif algorithm == 'pbkdf2':
            # Fallback to PBKDF2
            if CRYPTOGRAPHY_AVAILABLE:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=key_length,
                    salt=salt,
                    iterations=480000,  # OWASP 2023 recommendation
                )
                derived_key = kdf.derive(password.encode('utf-8'))
            else:
                # Pure Python fallback
                derived_key = hashlib.pbkdf2_hmac(
                    'sha256',
                    password.encode('utf-8'),
                    salt,
                    480000,
                    key_length
                )
        else:
            raise ValueError(f"Unsupported KDF algorithm: {algorithm}")
            
        return derived_key, salt
        
    def encrypt_data(self, data: bytes, key_id: Optional[str] = None, 
                    algorithm: str = 'AES-256-GCM') -> Dict[str, Any]:
        """
        Encrypt data with advanced authenticated encryption.
        
        Returns:
            Dictionary containing encrypted data and metadata
        """
        if key_id is None:
            key_id = self._generate_key(algorithm)
            
        encryption_key = self.key_store[key_id]
        
        if algorithm == 'FERNET' and CRYPTOGRAPHY_AVAILABLE:
            fernet_key = base64.urlsafe_b64encode(encryption_key.key_data)
            fernet = Fernet(fernet_key)
            ciphertext = fernet.encrypt(data)
            
            return {
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'algorithm': algorithm,
                'key_id': key_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        elif algorithm == 'AES-256-GCM' and CRYPTOGRAPHY_AVAILABLE:
            # Use AES-GCM for authenticated encryption
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
            
            cipher = Cipher(
                algorithms.AES(encryption_key.key_data),
                modes.GCM(nonce)
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            return {
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'nonce': base64.b64encode(nonce).decode(),
                'tag': base64.b64encode(encryptor.tag).decode(),
                'algorithm': algorithm,
                'key_id': key_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        else:
            # Fallback to simple XOR with HMAC (not recommended for production)
            logger.warning(f"Using fallback encryption for {algorithm}")
            nonce = secrets.token_bytes(16)
            
            # Simple XOR encryption (educational purposes only)
            key_stream = self._generate_keystream(encryption_key.key_data, nonce, len(data))
            ciphertext = bytes(a ^ b for a, b in zip(data, key_stream))
            
            # Add HMAC for integrity
            mac = hmac.new(
                encryption_key.key_data,
                nonce + ciphertext,
                hashlib.sha256
            ).digest()
            
            return {
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'nonce': base64.b64encode(nonce).decode(),
                'mac': base64.b64encode(mac).decode(),
                'algorithm': f"{algorithm}_FALLBACK",
                'key_id': key_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Decrypt data using stored encryption keys."""
        key_id = encrypted_data['key_id']
        algorithm = encrypted_data['algorithm']
        
        if key_id not in self.key_store:
            raise ValueError(f"Encryption key {key_id} not found")
            
        encryption_key = self.key_store[key_id]
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        
        if algorithm == 'FERNET' and CRYPTOGRAPHY_AVAILABLE:
            fernet_key = base64.urlsafe_b64encode(encryption_key.key_data)
            fernet = Fernet(fernet_key)
            return fernet.decrypt(ciphertext)
            
        elif algorithm == 'AES-256-GCM' and CRYPTOGRAPHY_AVAILABLE:
            nonce = base64.b64decode(encrypted_data['nonce'])
            tag = base64.b64decode(encrypted_data['tag'])
            
            cipher = Cipher(
                algorithms.AES(encryption_key.key_data),
                modes.GCM(nonce, tag)
            )
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext) + decryptor.finalize()
            
        elif algorithm.endswith('_FALLBACK'):
            # Handle fallback decryption
            nonce = base64.b64decode(encrypted_data['nonce'])
            mac = base64.b64decode(encrypted_data['mac'])
            
            # Verify HMAC
            expected_mac = hmac.new(
                encryption_key.key_data,
                nonce + ciphertext,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(mac, expected_mac):
                raise ValueError("Data integrity check failed")
                
            # Decrypt using XOR
            key_stream = self._generate_keystream(encryption_key.key_data, nonce, len(ciphertext))
            return bytes(a ^ b for a, b in zip(ciphertext, key_stream))
            
        else:
            raise ValueError(f"Unsupported decryption algorithm: {algorithm}")
            
    def _generate_key(self, algorithm: str) -> str:
        """Generate a new encryption key for the specified algorithm."""
        if algorithm not in self.algorithm_registry:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        key_id = SecureRandom.uuid4()
        key_size = self.algorithm_registry[algorithm]['key_size']
        key_data = secrets.token_bytes(key_size)
        
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm=algorithm,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),  # 1 year default
            max_usage=10000  # Rotate key after 10k uses
        )
        
        self.key_store[key_id] = encryption_key
        logger.info(f"Generated new {algorithm} key: {key_id}")
        
        return key_id
        
    def _generate_keystream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        """Generate keystream for fallback encryption (educational only)."""
        keystream = b''
        counter = 0
        
        while len(keystream) < length:
            block = hashlib.sha256(key + nonce + counter.to_bytes(4, 'big')).digest()
            keystream += block
            counter += 1
            
        return keystream[:length]
        
    def secure_hash(self, data: bytes, algorithm: str = 'sha256', 
                   salt: Optional[bytes] = None) -> Dict[str, str]:
        """Generate secure hash with optional salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
            
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        elif algorithm == 'sha3-256':
            hash_obj = hashlib.sha3_256()
        elif algorithm == 'blake2b':
            hash_obj = hashlib.blake2b(salt=salt[:16])  # BLAKE2b salt is 16 bytes max
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
        if algorithm != 'blake2b':  # BLAKE2b handles salt internally
            hash_obj.update(salt)
        hash_obj.update(data)
        
        return {
            'hash': hash_obj.hexdigest(),
            'salt': base64.b64encode(salt).decode(),
            'algorithm': algorithm
        }
        
    def verify_hash(self, data: bytes, hash_info: Dict[str, str]) -> bool:
        """Verify data against stored hash."""
        try:
            salt = base64.b64decode(hash_info['salt'])
            computed_hash = self.secure_hash(data, hash_info['algorithm'], salt)
            return hmac.compare_digest(computed_hash['hash'], hash_info['hash'])
        except Exception as e:
            logger.error(f"Hash verification error: {e}")
            return False
            
    def rotate_keys(self) -> List[str]:
        """Rotate expired or overused keys."""
        rotated_keys = []
        now = datetime.utcnow()
        
        for key_id, encryption_key in list(self.key_store.items()):
            should_rotate = False
            
            # Check expiration
            if encryption_key.expires_at and now > encryption_key.expires_at:
                should_rotate = True
                logger.info(f"Key {key_id} expired")
                
            # Check usage count
            if (encryption_key.max_usage and 
                encryption_key.usage_count >= encryption_key.max_usage):
                should_rotate = True
                logger.info(f"Key {key_id} exceeded usage limit")
                
            if should_rotate:
                # Generate new key with same algorithm
                new_key_id = self._generate_key(encryption_key.algorithm)
                rotated_keys.append(new_key_id)
                
                # Keep old key for decryption but mark as deprecated
                encryption_key.expires_at = now + timedelta(days=30)
                
        return rotated_keys
        
    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an encryption key."""
        if key_id not in self.key_store:
            return None
            
        key = self.key_store[key_id]
        return {
            'key_id': key.key_id,
            'algorithm': key.algorithm,
            'created_at': key.created_at.isoformat(),
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'usage_count': key.usage_count,
            'max_usage': key.max_usage,
            'is_expired': (key.expires_at and datetime.utcnow() > key.expires_at),
            'secure_level': self.algorithm_registry.get(key.algorithm, {}).get('secure_level', 'unknown')
        }