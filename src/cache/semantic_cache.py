# src/cache/semantic_cache.py
import hashlib
import json
import sqlite3
import numpy as np
from typing import Optional, List, Dict, Any, Union
from langchain.embeddings.base import Embeddings
import pickle
import time
from datetime import datetime, timedelta

class EnhancedSemanticCache:
    """Enhanced semantic cache with multiple caching strategies."""
    
    def __init__(self, 
                 db_path: str = "semantic_cache.db", 
                 similarity_threshold: float = 0.85,
                 ttl_hours: int = 24,
                 max_entries: int = 1000):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self.max_entries = max_entries
        self.embeddings = None
        self._init_db()
    
    def set_embeddings(self, embeddings: Embeddings):
        """Set the embedding model for semantic similarity."""
        self.embeddings = embeddings
    
    def _init_db(self):
        """Initialize the cache database with enhanced schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT,
                    query_text TEXT,
                    response TEXT,
                    embedding BLOB,
                    similarity_used REAL,
                    hit_count INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create index for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON semantic_cache(query_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON semantic_cache(created_at)")
            
            conn.commit()
    
    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a query with error handling."""
        if not self.embeddings:
            return None
        
        try:
            embedding = self.embeddings.embed_query(query)
            return np.array(embedding, dtype=np.float32)  # Use float32 for memory efficiency
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get(self, query: str, return_metadata: bool = False) -> Union[Optional[str], Optional[Dict[str, Any]]]:
        """Get cached response for a query using semantic similarity."""
        if not self.embeddings:
            return None
        
        # Clean expired entries first
        self._clean_expired()
        
        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, query_text, response, embedding, hit_count, metadata 
                FROM semantic_cache 
                WHERE datetime(created_at, '+' || ? || ' hours') > datetime('now')
            """, (self.ttl_hours,))
            
            for row in cursor.fetchall():
                entry_id, cached_query, cached_response, cached_embedding_blob, hit_count, metadata_str = row
                
                if cached_embedding_blob:
                    try:
                        cached_embedding = np.frombuffer(cached_embedding_blob, dtype=np.float32)
                        similarity = self._cosine_similarity(query_embedding, cached_embedding)
                        
                        if similarity >= self.similarity_threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                "id": entry_id,
                                "response": cached_response,
                                "similarity": similarity,
                                "hit_count": hit_count,
                                "cached_query": cached_query,
                                "metadata": json.loads(metadata_str) if metadata_str else {}
                            }
                    except Exception as e:
                        print(f"Error processing cached embedding: {e}")
                        continue
        
        if best_match:
            # Update hit count and last accessed
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE semantic_cache 
                    SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (best_match["id"],))
                conn.commit()
            
            print(f"Semantic cache hit! Similarity: {best_similarity:.3f}, Hit count: {best_match['hit_count'] + 1}")
            
            if return_metadata:
                return best_match
            else:
                return best_match["response"]
        
        return None
    
    def set(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Cache a query-response pair with embedding and metadata."""
        if not self.embeddings:
            return
        
        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            return
        
        # Check if we need to make room
        self._enforce_size_limit()
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        embedding_blob = query_embedding.astype(np.float32).tobytes()
        metadata_str = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if similar query already exists
            existing = self.get(query, return_metadata=True)
            if existing and existing["similarity"] > 0.95:  # Very high similarity
                # Update existing entry instead of creating duplicate
                conn.execute("""
                    UPDATE semantic_cache 
                    SET response = ?, hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (response, existing["id"]))
            else:
                # Insert new entry
                conn.execute("""
                    INSERT INTO semantic_cache 
                    (query_hash, query_text, response, embedding, metadata) 
                    VALUES (?, ?, ?, ?, ?)
                """, (query_hash, query, response, embedding_blob, metadata_str))
            
            conn.commit()
    
    def _clean_expired(self):
        """Remove expired cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM semantic_cache 
                WHERE datetime(created_at, '+' || ? || ' hours') <= datetime('now')
            """, (self.ttl_hours,))
            conn.commit()
    
    def _enforce_size_limit(self):
        """Enforce maximum cache size by removing least recently used entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM semantic_cache")
            current_size = cursor.fetchone()[0]
            
            if current_size >= self.max_entries:
                # Remove oldest entries (by last_accessed)
                entries_to_remove = current_size - self.max_entries + 100  # Remove extra to avoid frequent cleanup
                conn.execute("""
                    DELETE FROM semantic_cache 
                    WHERE id IN (
                        SELECT id FROM semantic_cache 
                        ORDER BY last_accessed ASC 
                        LIMIT ?
                    )
                """, (entries_to_remove,))
                conn.commit()
                print(f"Removed {entries_to_remove} old cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute("SELECT COUNT(*) FROM semantic_cache")
            total_entries = cursor.fetchone()[0]
            
            # Total hits
            cursor.execute("SELECT SUM(hit_count) FROM semantic_cache")
            total_hits = cursor.fetchone()[0] or 0
            
            # Average similarity of cached entries
            cursor.execute("SELECT AVG(similarity_used) FROM semantic_cache WHERE similarity_used IS NOT NULL")
            avg_similarity = cursor.fetchone()[0] or 0
            
            # Most hit entry
            cursor.execute("SELECT query_text, hit_count FROM semantic_cache ORDER BY hit_count DESC LIMIT 1")
            most_hit = cursor.fetchone()
            
            return {
                "total_entries": total_entries,
                "total_hits": total_hits,
                "average_similarity": avg_similarity,
                "most_hit_query": most_hit[0] if most_hit else None,
                "most_hit_count": most_hit[1] if most_hit else 0,
                "cache_efficiency": total_hits / max(total_entries, 1)
            }
    
    def clear(self):
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM semantic_cache")
            conn.commit()
    
    def size(self) -> int:
        """Get the number of cached entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM semantic_cache")
            return cursor.fetchone()[0]

class EmbeddingCache:
    """Cache for embedding computations to avoid recomputing same embeddings."""
    
    def __init__(self, db_path: str = "embedding_cache.db", max_entries: int = 5000):
        self.db_path = db_path
        self.max_entries = max_entries
        self._init_db()
    
    def _init_db(self):
        """Initialize embedding cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT UNIQUE,
                    text_content TEXT,
                    embedding BLOB,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_text_hash ON embedding_cache(text_hash)")
            conn.commit()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embedding_cache WHERE text_hash = ?", (text_hash,))
            result = cursor.fetchone()
            
            if result:
                # Update access count
                conn.execute("UPDATE embedding_cache SET access_count = access_count + 1 WHERE text_hash = ?", (text_hash,))
                conn.commit()
                
                return np.frombuffer(result[0], dtype=np.float32)
        
        return None
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache an embedding for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        # Enforce size limit
        self._enforce_size_limit()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embedding_cache (text_hash, text_content, embedding) 
                VALUES (?, ?, ?)
            """, (text_hash, text, embedding_blob))
            conn.commit()
    
    def _enforce_size_limit(self):
        """Enforce maximum cache size."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embedding_cache")
            current_size = cursor.fetchone()[0]
            
            if current_size >= self.max_entries:
                # Remove least accessed entries
                entries_to_remove = current_size - self.max_entries + 100
                conn.execute("""
                    DELETE FROM embedding_cache 
                    WHERE id IN (
                        SELECT id FROM embedding_cache 
                        ORDER BY access_count ASC, created_at ASC 
                        LIMIT ?
                    )
                """, (entries_to_remove,))
                conn.commit()

# Legacy compatibility
SemanticCache = EnhancedSemanticCache
