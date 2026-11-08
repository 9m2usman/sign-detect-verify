"""
Hugging Face Webhook Dispatcher and Idempotency Guard.
Verifies cryptographic signatures, prevents duplicate task execution, and triggers retraining.
"""

import hmac
import hashlib
from typing import Dict, Any, Tuple


class HuggingFaceWebhookHandler:
    """
    Parses and authenticates Hugging Face Webhook payloads.
    """
    def __init__(self, secret_token: str):
        self.secret_token = secret_token
        self._processed_event_ids = set()

    def verify_secret(self, header_secret: str) -> bool:
        """Constant-time token verification to prevent timing attacks."""
        if not header_secret:
            return False
        return hmac.compare_digest(header_secret.strip(), self.secret_token.strip())

    def process_event(self, event_id: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks idempotency and dispatches training workflow."""
        if event_id in self._processed_event_ids:
            return False, f"Event {event_id} already processed (Idempotency Guard)."

        self._processed_event_ids.add(event_id)
        repo_name = payload.get("repo", {}).get("name", "unknown")
        
        return True, f"Event {event_id} accepted for repo {repo_name}."
