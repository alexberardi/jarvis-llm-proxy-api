import logging
import time
import threading
from typing import List, Dict, Optional, Any
from .conversation_cache import ConversationCache

logger = logging.getLogger("uvicorn")

class LocalConversationCache(ConversationCache):
    """In-memory conversation cache with automatic cleanup"""
    
    def __init__(self, ttl_seconds: int = 600, cleanup_interval: int = 30):
        self.cache: Dict[str, Dict] = {}
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self.lock = threading.Lock()
        
        # Start background cleanup task
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get_session(self, conversation_id: str) -> Optional[Dict]:
        """Get session data if it exists and is not expired"""
        with self.lock:
            if conversation_id not in self.cache:
                return None
            
            session = self.cache[conversation_id]
            current_time = time.time()
            
            # Check if expired
            if current_time - session["created_at"] > self.ttl_seconds:
                # Remove expired session
                del self.cache[conversation_id]
                return None
            
            # Update last accessed time
            session["last_accessed"] = current_time
            return session.copy()  # Return copy to prevent external modification
    
    def create_or_update_session(self, conversation_id: str, messages: List[Dict]) -> bool:
        """Create new session or update existing one with new messages"""
        current_time = time.time()
        
        with self.lock:
            if conversation_id in self.cache:
                # Update existing session - append new messages
                existing_session = self.cache[conversation_id]
                existing_messages = existing_session["messages"]
                
                # Find the last message that's not in the existing conversation
                # This handles the case where caller sends full conversation
                if len(messages) > len(existing_messages):
                    # Append only the new messages
                    new_messages = messages[len(existing_messages):]
                    existing_messages.extend(new_messages)
                    logger.debug(f"🔄 Updated session {conversation_id} with {len(new_messages)} new messages")
                else:
                    # No new messages to append
                    logger.debug(f"🔄 Session {conversation_id} updated (no new messages)")
                
                # Update timestamps
                existing_session["last_accessed"] = current_time
            else:
                # Create new session
                self.cache[conversation_id] = {
                    "messages": messages.copy(),
                    "created_at": current_time,
                    "last_accessed": current_time,
                    "warmup_status": "pending"  # Initialize warm-up status
                }
                logger.debug(f"🆕 Created new session {conversation_id} with {len(messages)} messages")
            
            return True
    
    def delete_session(self, conversation_id: str) -> bool:
        """Delete a specific session"""
        with self.lock:
            if conversation_id in self.cache:
                del self.cache[conversation_id]
                logger.debug(f"🗑️  Deleted session {conversation_id}")
                return True
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions, return count of removed sessions"""
        current_time = time.time()
        expired_sessions = []
        
        with self.lock:
            for conversation_id, session in self.cache.items():
                if current_time - session["created_at"] > self.ttl_seconds:
                    expired_sessions.append(conversation_id)
            
            # Remove expired sessions
            for conversation_id in expired_sessions:
                del self.cache[conversation_id]
        
        if expired_sessions:
            logger.debug(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_session_count(self) -> int:
        """Get current number of active sessions"""
        with self.lock:
            return len(self.cache)
    
    def update_processed_context(self, conversation_id: str, processed_context: Any) -> bool:
        """Update session with processed context from LLM"""
        with self.lock:
            if conversation_id in self.cache:
                self.cache[conversation_id]["processed_context"] = processed_context
                logger.debug(f"💾 Updated processed context for session {conversation_id}")
                return True
            return False
    
    def get_processed_context(self, conversation_id: str) -> Optional[Any]:
        """Get processed context if available"""
        with self.lock:
            if conversation_id in self.cache:
                return self.cache[conversation_id].get("processed_context")
            return None
    
    def get_warmup_status(self, conversation_id: str) -> Optional[str]:
        """Get warm-up status: 'pending', 'in_progress', 'completed', or None if no session"""
        with self.lock:
            if conversation_id in self.cache:
                return self.cache[conversation_id].get("warmup_status")
            return None
    
    def set_warmup_status(self, conversation_id: str, status: str) -> bool:
        """Set warm-up status for a session"""
        with self.lock:
            if conversation_id in self.cache:
                self.cache[conversation_id]["warmup_status"] = status
                logger.debug(f"🔄 Updated warm-up status for session {conversation_id}: {status}")
                return True
            return False
    
    def _cleanup_loop(self):
        """Background thread that runs cleanup every cleanup_interval seconds"""
        while True:
            time.sleep(self.cleanup_interval)
            try:
                removed_count = self.cleanup_expired()
                if removed_count > 0:
                    logger.debug(f"🔄 Background cleanup removed {removed_count} expired sessions")
            except Exception as e:
                logger.warning(f"⚠️  Error in background cleanup: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread.is_alive():
            # Note: daemon threads are automatically terminated when main thread exits
            pass
