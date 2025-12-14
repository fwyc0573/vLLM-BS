# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Transfer Synchronization Module.

This module provides synchronization mechanisms for disaggregated
prefill-decode architecture, ensuring decode phase starts only after
prefill phase completes KV cache transfer.

Usage:
    Prefill side:
        from vllm.request_generator.kv_sync import KVTransferSync
        sync = KVTransferSync(storage_path="local_storage")
        # ... run prefill ...
        sync.mark_transfer_complete()
    
    Decode side:
        from vllm.request_generator.kv_sync import KVTransferSync
        sync = KVTransferSync(storage_path="local_storage")
        sync.wait_for_transfer_complete(timeout=60)
        # ... run decode ...
"""

import os
import time
import logging
from typing import Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class KVTransferSync:
    """Synchronization helper for KV cache transfer between prefill and decode.
    
    In disaggregated prefill-decode architecture, the decode cluster needs to
    wait for the prefill cluster to complete KV cache transfer before starting.
    This class provides mechanisms to coordinate this synchronization.
    
    Synchronization is achieved through marker files:
    - `.kv_transfer_complete` indicates successful completion
    - `.kv_transfer_error` indicates an error occurred
    
    Attributes:
        storage_path: Path to the shared KV cache storage directory.
        marker_file: Path to the completion marker file.
        error_file: Path to the error marker file.
    """
    
    COMPLETE_MARKER = ".kv_transfer_complete"
    ERROR_MARKER = ".kv_transfer_error"
    
    def __init__(
        self,
        storage_path: str = "local_storage",
        create_dirs: bool = True,
    ):
        """Initialize the KV transfer synchronization helper.
        
        Args:
            storage_path: Path to the shared KV cache storage directory.
            create_dirs: If True, create the storage directory if it doesn't exist.
        """
        self.storage_path = Path(storage_path)
        
        if create_dirs and not self.storage_path.exists():
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created KV storage directory: {self.storage_path}")
        
        self.marker_file = self.storage_path / self.COMPLETE_MARKER
        self.error_file = self.storage_path / self.ERROR_MARKER
    
    def mark_transfer_complete(self, metadata: Optional[dict] = None) -> None:
        """Mark KV transfer as complete.
        
        Should be called by the prefill cluster after successfully completing
        KV cache transfer to shared storage.
        
        Args:
            metadata: Optional dict with transfer metadata (e.g., num_requests,
                     timestamp, etc.). Will be written to the marker file.
        """
        import json
        from datetime import datetime
        
        # Remove any existing error marker
        if self.error_file.exists():
            self.error_file.unlink()
        
        # Write completion marker with metadata
        marker_content = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "storage_path": str(self.storage_path),
        }
        if metadata:
            marker_content.update(metadata)
        
        with open(self.marker_file, "w") as f:
            json.dump(marker_content, f, indent=2)
        
        logger.info(f"KV transfer marked complete: {self.marker_file}")
    
    def mark_transfer_error(self, error_message: str) -> None:
        """Mark KV transfer as failed.
        
        Should be called by the prefill cluster if an error occurs during
        KV cache transfer.
        
        Args:
            error_message: Description of the error.
        """
        import json
        from datetime import datetime
        
        error_content = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
        }
        
        with open(self.error_file, "w") as f:
            json.dump(error_content, f, indent=2)
        
        logger.error(f"KV transfer error: {error_message}")
    
    def is_transfer_complete(self) -> bool:
        """Check if KV transfer is complete.
        
        Returns:
            True if the completion marker exists and no error marker exists.
        """
        if self.error_file.exists():
            return False
        return self.marker_file.exists()
    
    def has_transfer_error(self) -> Optional[str]:
        """Check if KV transfer encountered an error.
        
        Returns:
            Error message if error marker exists, None otherwise.
        """
        if not self.error_file.exists():
            return None
        
        import json
        try:
            with open(self.error_file, "r") as f:
                content = json.load(f)
                return content.get("error", "Unknown error")
        except Exception as e:
            return f"Error reading error file: {e}"
    
    def wait_for_transfer_complete(
        self,
        timeout: float = 60.0,
        poll_interval: float = 0.5,
        verbose: bool = True,
    ) -> bool:
        """Wait for KV transfer to complete.
        
        Polls the completion marker file until it exists or timeout is reached.
        Should be called by the decode cluster before starting decode operations.
        
        Args:
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between polls in seconds.
            verbose: If True, log progress messages.
            
        Returns:
            True if transfer completed successfully within timeout.
            
        Raises:
            TimeoutError: If timeout is reached before transfer completes.
            RuntimeError: If transfer encountered an error.
        """
        start_time = time.time()
        last_log_time = start_time
        
        if verbose:
            logger.info(f"Waiting for KV transfer to complete (timeout={timeout}s)...")
            print(f"[KV Sync] Waiting for KV transfer to complete (timeout={timeout}s)...")
        
        while True:
            # Check for error
            error_msg = self.has_transfer_error()
            if error_msg:
                raise RuntimeError(f"KV transfer failed: {error_msg}")
            
            # Check for completion
            if self.is_transfer_complete():
                elapsed = time.time() - start_time
                if verbose:
                    logger.info(f"KV transfer complete after {elapsed:.2f}s")
                    print(f"[KV Sync] KV transfer complete after {elapsed:.2f}s")
                return True
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timeout waiting for KV transfer after {timeout}s. "
                    f"Marker file not found: {self.marker_file}"
                )
            
            # Log progress periodically (every 5 seconds)
            if verbose and time.time() - last_log_time >= 5.0:
                remaining = timeout - elapsed
                print(f"[KV Sync] Still waiting... ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining)")
                last_log_time = time.time()
            
            # Sleep before next poll
            time.sleep(poll_interval)
    
    def cleanup(self) -> None:
        """Remove all marker files.
        
        Should be called before starting a new prefill-decode cycle
        to ensure clean state.
        """
        if self.marker_file.exists():
            self.marker_file.unlink()
            logger.info(f"Removed completion marker: {self.marker_file}")
        
        if self.error_file.exists():
            self.error_file.unlink()
            logger.info(f"Removed error marker: {self.error_file}")
    
    def get_transfer_metadata(self) -> Optional[dict]:
        """Get metadata from the completion marker file.
        
        Returns:
            Metadata dict if marker exists, None otherwise.
        """
        if not self.marker_file.exists():
            return None
        
        import json
        try:
            with open(self.marker_file, "r") as f:
                return json.load(f)
        except Exception:
            return None


def create_kv_sync(
    storage_path: str = "local_storage",
    kv_transfer_config: Optional["KVTransferConfig"] = None,  # noqa: F821
) -> KVTransferSync:
    """Create a KVTransferSync instance from config.
    
    Args:
        storage_path: Default storage path if not in config.
        kv_transfer_config: Optional KVTransferConfig to extract path from.
        
    Returns:
        Configured KVTransferSync instance.
    """
    # Try to extract storage path from KVTransferConfig
    if kv_transfer_config is not None:
        extra_config = kv_transfer_config.kv_connector_extra_config or {}
        if "shared_storage_path" in extra_config:
            storage_path = extra_config["shared_storage_path"]
    
    return KVTransferSync(storage_path=storage_path)
