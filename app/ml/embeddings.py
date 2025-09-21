"""
Embeddings generation for startup similarity analysis.
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for startup similarity matching."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def generate_startup_embedding(self, startup_description: str, pitch_content: str = "") -> np.ndarray:
        """Generate embedding for a startup."""
        if not self.model:
            # Return zero vector if model not loaded
            return np.zeros(self.embedding_dim)
        
        try:
            # Combine description and pitch content
            combined_text = f"{startup_description} {pitch_content}".strip()
            
            if not combined_text:
                return np.zeros(self.embedding_dim)
            
            # Generate embedding
            embedding = self.model.encode(combined_text)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def find_similar_startups(
        self, 
        startup_embedding: np.ndarray, 
        candidate_embeddings: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most similar startups based on embeddings."""
        if not self.model or len(candidate_embeddings) == 0:
            return []
        
        try:
            similarities = []
            
            for candidate in candidate_embeddings:
                if "embedding" not in candidate:
                    continue
                
                candidate_emb = np.array(candidate["embedding"])
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(startup_embedding, candidate_emb)
                
                similarities.append({
                    **candidate,
                    "similarity_score": similarity
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar startups: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0