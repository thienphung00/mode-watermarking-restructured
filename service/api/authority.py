"""
Key registration and policy management.

The Authority is responsible for:
- Key registration and validation
- Deriving scoped keys for GPU workers
- Managing embedding and detection configurations
- Computing policy versions for consistency
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from enum import Enum
from typing import Any, Dict, Optional

from service.api.key_store import get_key_store

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Operation types for scoped key derivation."""
    GENERATION = "generation"
    DETECTION = "detection"


def derive_scoped_key(
    master_key: str,
    key_id: str,
    operation: OperationType,
    request_id: Optional[str] = None,
) -> str:
    """
    Derive a scoped ephemeral key from the master key.
    
    SECURITY:
    - Master key never leaves the API boundary
    - Workers only receive derived keys
    - Derived keys are scoped to specific operations
    
    Args:
        master_key: The master key (hex string)
        key_id: Public key identifier
        operation: Operation type (generation or detection)
        request_id: Optional request ID for logging
        
    Returns:
        64-character hex derived key
    """
    context = f"watermark_derived_key_v1:{operation.value}:{key_id}"
    
    # HKDF-like construction with HMAC-SHA256
    extract_key = hmac.new(
        key=b"watermark_extract_salt_v1",
        msg=bytes.fromhex(master_key),
        digestmod=hashlib.sha256,
    ).digest()
    
    derived_bytes = hmac.new(
        key=extract_key,
        msg=context.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    
    return derived_bytes.hex()


class Authority:
    """
    Watermark authority for key management and policy.
    
    Responsibilities:
    - Key validation
    - Scoped key derivation
    - Embedding configuration
    - Detection configuration
    """
    
    # Default embedding configuration (seed-bias watermarking)
    DEFAULT_EMBEDDING_CONFIG = {
        "lambda_strength": 0.05,
        "domain": "frequency",
        "low_freq_cutoff": 0.05,
        "high_freq_cutoff": 0.4,
    }
    
    # Default detection configuration
    DEFAULT_DETECTION_CONFIG = {
        "detector_type": "bayesian",
        "threshold": 0.5,
        "prior_watermarked": 0.5,
    }
    
    # Default G-field configuration
    DEFAULT_G_FIELD_CONFIG = {
        "mapping_mode": "binary",
        "domain": "frequency",
        "frequency_mode": "bandpass",
        "low_freq_cutoff": 0.05,
        "high_freq_cutoff": 0.4,
        "normalize_zero_mean": True,
        "normalize_unit_variance": True,
    }
    
    # Default inversion configuration
    DEFAULT_INVERSION_CONFIG = {
        "num_inference_steps": 50,
        "guidance_scale": 1.0,
    }
    
    def __init__(self):
        """Initialize authority."""
        self.key_store = get_key_store()
    
    def validate_key(self, key_id: str) -> bool:
        """
        Validate that a key exists and is active.
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if valid and active
        """
        return self.key_store.is_active(key_id)
    
    def get_generation_payload(
        self,
        key_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Get payload for generation request to GPU worker.
        
        SECURITY: Returns derived_key, never master_key.
        
        Args:
            key_id: Key identifier
            request_id: Request ID for tracing
            
        Returns:
            Dictionary with derived_key, fingerprint, embedding_config
            
        Raises:
            ValueError: If key not found or inactive
        """
        master_key = self.key_store.get_master_key(key_id)
        if master_key is None:
            raise ValueError(f"Key {key_id} not found or inactive")
        
        fingerprint = self.key_store.get_fingerprint(key_id)
        derived_key = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.GENERATION,
            request_id=request_id,
        )
        
        return {
            "key_id": key_id,
            "derived_key": derived_key,
            "key_fingerprint": fingerprint,
            "embedding_config": self.DEFAULT_EMBEDDING_CONFIG.copy(),
        }
    
    def get_detection_payload(
        self,
        key_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Get payload for detection request to GPU worker.
        
        SECURITY: Returns derived_key, never master_key.
        
        Args:
            key_id: Key identifier
            request_id: Request ID for tracing
            
        Returns:
            Dictionary with derived_key, fingerprint, detection configs
            
        Raises:
            ValueError: If key not found or inactive
        """
        master_key = self.key_store.get_master_key(key_id)
        if master_key is None:
            raise ValueError(f"Key {key_id} not found or inactive")
        
        fingerprint = self.key_store.get_fingerprint(key_id)
        derived_key = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.DETECTION,
            request_id=request_id,
        )
        
        return {
            "key_id": key_id,
            "derived_key": derived_key,
            "key_fingerprint": fingerprint,
            "g_field_config": self.DEFAULT_G_FIELD_CONFIG.copy(),
            "detection_config": self.DEFAULT_DETECTION_CONFIG.copy(),
            "inversion_config": self.DEFAULT_INVERSION_CONFIG.copy(),
        }
    
    @staticmethod
    def compute_policy_version(
        embedding_config: Dict[str, Any],
        detection_config: Dict[str, Any],
    ) -> str:
        """
        Compute deterministic policy version from configurations.
        
        Args:
            embedding_config: Embedding configuration
            detection_config: Detection configuration
            
        Returns:
            16-character policy version hash
        """
        policy_dict = {
            "embedding": dict(sorted(embedding_config.items())),
            "detection": dict(sorted(detection_config.items())),
        }
        policy_json = json.dumps(policy_dict, sort_keys=True, separators=(',', ':'))
        hash_hex = hashlib.sha256(policy_json.encode('utf-8')).hexdigest()
        return hash_hex[:16]


# Global authority instance
_authority: Optional[Authority] = None


def get_authority() -> Authority:
    """Get the global authority instance."""
    global _authority
    if _authority is None:
        _authority = Authority()
    return _authority


def reset_authority() -> None:
    """Reset the global authority (useful for testing)."""
    global _authority
    _authority = None
