import os
from typing import Optional, Any
from .conversation_cache import ConversationCache
from .local_conversation_cache import LocalConversationCache

class CacheManager:
    """Manages conversation cache implementation based on environment configuration"""
    
    def __init__(self):
        self.cache: Optional[ConversationCache] = None
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache based on environment variables"""
        cache_type = os.getenv("JARVIS_CACHE_TYPE", "local").lower()
        ttl_seconds = int(os.getenv("JARVIS_SESSION_TTL", "600"))  # 10 minutes default
        cleanup_interval = int(os.getenv("JARVIS_CACHE_CLEANUP_INTERVAL", "30"))  # 30 seconds default
        
        print(f"🔧 Initializing cache: type={cache_type}, ttl={ttl_seconds}s, cleanup_interval={cleanup_interval}s")
        
        if cache_type == "local":
            self.cache = LocalConversationCache(ttl_seconds, cleanup_interval)
            print("✅ Local in-memory cache initialized")
        elif cache_type == "redis":
            # Future Redis implementation
            print("⚠️  Redis cache not yet implemented, falling back to local")
            self.cache = LocalConversationCache(ttl_seconds, cleanup_interval)
        else:
            print(f"⚠️  Unknown cache type '{cache_type}', falling back to local")
            self.cache = LocalConversationCache(ttl_seconds, cleanup_interval)
    
    def get_cache(self) -> ConversationCache:
        """Get the current cache instance"""
        if self.cache is None:
            raise RuntimeError("Cache not initialized")
        return self.cache
    
    def get_session_count(self) -> int:
        """Get current number of active sessions"""
        return self.cache.get_session_count() if self.cache else 0
