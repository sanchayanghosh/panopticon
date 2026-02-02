import hashlib
import json
import uuid
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import torch
except ImportError:
    torch = None

class MerkleLogger:
    def __init__(self):
        self.ledger: List[Dict[str, Any]] = []
        self.hardware_id = self._get_hardware_id()
        self.firmware_signature = self._get_firmware_signature()

    def _get_hardware_id(self) -> str:
        """
        Retrieves a unique hardware identifier. 
        In production, this should interface with TPM or Secure Enclave.
        """
        # Using MAC address as a proxy for hardware ID
        mac_num = uuid.getnode()
        mac = ':'.join(('%012X' % mac_num)[i:i+2] for i in range(0, 12, 2))
        return hashlib.sha256(f"HW_ID_{mac}".encode()).hexdigest()

    def _get_firmware_signature(self) -> str:
        """
        Simulates retrieving a firmware cryptographic signature.
        """
        # Placeholder for actual firmware attestation
        system_info = f"{platform.system()}_{platform.release()}_{platform.version()}"
        return hashlib.sha256(f"FW_SIG_{system_info}".encode()).hexdigest()

    def hash_tensor(self, tensor: Any) -> str:
        """
        Hashes a PyTorch tensor.
        """
        if torch and isinstance(tensor, torch.Tensor):
            return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
        # Fallback for non-tensor objects or if torch is missing
        return hashlib.sha256(str(tensor).encode()).hexdigest()

    def log_step(self, epoch: int, batch_data: Any, model_weights: Any) -> Dict[str, Any]:
        """
        Logs a training step into the Merkle Tree Ledger.
        """
        # 1. Fingerprint the Data
        # Ensure batch_data is serializable; if not, use string representation
        try:
            data_str = json.dumps(batch_data, sort_keys=True)
        except (TypeError, ValueError):
            data_str = str(batch_data)
        
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        # 2. Fingerprint the Model State
        weight_hash = self.hash_tensor(model_weights)
        
        # 3. Create Block
        prev_hash = self.ledger[-1]['hash'] if self.ledger else "GENESIS_BLOCK_HASH_000000"
        
        block = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "data_fingerprint": data_hash,
            "weight_fingerprint": weight_hash,
            "hardware_id": self.hardware_id,
            "firmware_signature": self.firmware_signature,
            "prev_block_hash": prev_hash
        }
        
        # 4. Seal the Block (Proof of History)
        # We hash the entire block content including the previous hash
        block_content = json.dumps(block, sort_keys=True).encode()
        block['hash'] = hashlib.sha256(block_content).hexdigest()
        
        self.ledger.append(block)
        return block

    def verify_integrity(self) -> bool:
        """
        Re-computes hashes to verify the chain has not been tampered with.
        """
        for i, block in enumerate(self.ledger):
            # Reconstruct the block content for hashing
            # Note: We must exclude the 'hash' field itself to check validity
            content_to_hash = {k: v for k, v in block.items() if k != 'hash'}
            calculated_hash = hashlib.sha256(json.dumps(content_to_hash, sort_keys=True).encode()).hexdigest()
            
            if calculated_hash != block['hash']:
                print(f"Integrity Breach at Block {i}: Hash Mismatch")
                return False
                
            if i > 0:
                if block['prev_block_hash'] != self.ledger[i-1]['hash']:
                    print(f"Integrity Breach at Block {i}: Broken Chain Link")
                    return False
                    
        return True

    def get_provenance(self, block_index: int) -> Dict[str, Any]:
        if 0 <= block_index < len(self.ledger):
            return self.ledger[block_index]
        return {}
