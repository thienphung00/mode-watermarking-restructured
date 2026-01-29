"""
Persistent key registry using JSON storage.

Manages watermark keys with:
- Unique key ID generation
- Master key storage (encrypted in production)
- Fingerprint computation
- Key metadata
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KeyStore:
    """
    JSON-based persistent key storage.
    
    Keys are stored with:
    - key_id: Public identifier
    - master_key: Secret key (hex string)
    - fingerprint: Derived fingerprint for validation
    - created_at: ISO timestamp
    - metadata: Optional user metadata
    - is_active: Whether key is active
    """
    
    def __init__(self, store_path: str = "./data/keys.json"):
        """
        Initialize key store.
        
        Args:
            store_path: Path to JSON file for persistence
        """
        self.store_path = Path(store_path)
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self) -> None:
        """Load keys from persistent storage."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    data = json.load(f)
                    self._keys = data.get("keys", {})
                logger.info(f"Loaded {len(self._keys)} keys from {self.store_path}")
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
                self._keys = {}
        else:
            # Ensure parent directory exists
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self._keys = {}
            self._save()
    
    def _save(self) -> None:
        """Save keys to persistent storage."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "w") as f:
                json.dump({"keys": self._keys}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    @staticmethod
    def _generate_key_id() -> str:
        """Generate a unique key identifier."""
        random_bytes = secrets.token_bytes(8)
        hex_str = random_bytes.hex()[:10]
        return f"wm_{hex_str}"
    
    @staticmethod
    def _generate_master_key() -> str:
        """Generate a cryptographically secure master key."""
        return secrets.token_bytes(32).hex()
    
    @staticmethod
    def _compute_fingerprint(master_key: str) -> str:
        """
        Compute deterministic fingerprint from master key.
        
        The fingerprint is non-reversible but deterministic.
        """
        fingerprint_bytes = hmac.new(
            key=b"watermark_fingerprint_v1",
            msg=master_key.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        return fingerprint_bytes.hex()[:32]
    
    def register_key(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new watermark key.
        
        Args:
            metadata: Optional metadata to associate with key
            
        Returns:
            Dictionary with key_id, fingerprint, created_at
        """
        key_id = self._generate_key_id()
        master_key = self._generate_master_key()
        fingerprint = self._compute_fingerprint(master_key)
        created_at = datetime.now(timezone.utc).isoformat()
        
        key_record = {
            "key_id": key_id,
            "master_key": master_key,
            "fingerprint": fingerprint,
            "created_at": created_at,
            "metadata": metadata or {},
            "is_active": True,
        }
        
        self._keys[key_id] = key_record
        self._save()
        
        logger.info(f"Registered new key: {key_id}")
        
        return {
            "key_id": key_id,
            "fingerprint": fingerprint,
            "created_at": created_at,
        }
    
    def get_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """
        Get key record by ID.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Key record or None if not found
        """
        return self._keys.get(key_id)
    
    def get_master_key(self, key_id: str) -> Optional[str]:
        """
        Get master key for a key ID.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Master key string or None if not found
        """
        record = self._keys.get(key_id)
        if record and record.get("is_active", True):
            return record.get("master_key")
        return None
    
    def get_fingerprint(self, key_id: str) -> Optional[str]:
        """
        Get fingerprint for a key ID.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Fingerprint string or None if not found
        """
        record = self._keys.get(key_id)
        if record:
            return record.get("fingerprint")
        return None
    
    def is_active(self, key_id: str) -> bool:
        """Check if a key is active."""
        record = self._keys.get(key_id)
        return record is not None and record.get("is_active", True)
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke (deactivate) a key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if revoked, False if not found
        """
        if key_id in self._keys:
            self._keys[key_id]["is_active"] = False
            self._save()
            logger.info(f"Revoked key: {key_id}")
            return True
        return False
    
    def list_keys(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        List all registered keys.
        
        Args:
            include_inactive: Whether to include revoked keys
            
        Returns:
            List of key info dictionaries (without master_key)
        """
        result = []
        for key_id, record in self._keys.items():
            if not include_inactive and not record.get("is_active", True):
                continue
            result.append({
                "key_id": record["key_id"],
                "fingerprint": record["fingerprint"],
                "created_at": record["created_at"],
                "metadata": record.get("metadata", {}),
                "is_active": record.get("is_active", True),
            })
        return result
    
    def count(self) -> int:
        """Get total number of active keys."""
        return sum(1 for r in self._keys.values() if r.get("is_active", True))


# Global key store instance
_key_store: Optional[KeyStore] = None


def get_key_store() -> KeyStore:
    """Get the global key store instance."""
    global _key_store
    if _key_store is None:
        from service.api.config import get_config
        config = get_config()
        _key_store = KeyStore(store_path=config.key_store_path)
    return _key_store


def reset_key_store() -> None:
    """Reset the global key store (useful for testing)."""
    global _key_store
    _key_store = None
