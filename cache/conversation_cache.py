from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import time
import uuid

class ConversationCache(ABC):
    """Abstract base class for conversation session caching"""
    
    @abstractmethod
    def get_session(self, conversation_id: str) -> Optional[Dict]:
        """Get session data if it exists and is not expired"""
        pass
    
    @abstractmethod
    def create_or_update_session(self, conversation_id: str, messages: List[Dict]) -> bool:
        """Create new session or update existing one with new messages"""
        pass
    
    @abstractmethod
    def delete_session(self, conversation_id: str) -> bool:
        """Delete a specific session"""
        pass
    
    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired sessions, return count of removed sessions"""
        pass
    
    @abstractmethod
    def get_session_count(self) -> int:
        """Get current number of active sessions"""
        pass
    
    @abstractmethod
    def update_processed_context(self, conversation_id: str, processed_context: Any) -> bool:
        """Update session with processed context from LLM"""
        pass
    
    @abstractmethod
    def get_processed_context(self, conversation_id: str) -> Optional[Any]:
        """Get processed context if available"""
        pass
    
    @abstractmethod
    def get_warmup_status(self, conversation_id: str) -> Optional[str]:
        """Get warm-up status: 'pending', 'in_progress', 'completed', or None if no session"""
        pass
    
    @abstractmethod
    def set_warmup_status(self, conversation_id: str, status: str) -> bool:
        """Set warm-up status for a session"""
        pass
