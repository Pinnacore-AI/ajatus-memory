#!/usr/bin/env python3
# Ajatuskumppani — built in Finland, by the free minds of Pinnacore.

"""
AjatusMemory Embedding Generator

Generates vector embeddings for user interactions and memories.
"""

from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded embedding model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def generate(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: A single text string or list of text strings
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1, emb2 = self.generate([text1, text2])
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embeddings"""
        return self.model.get_sentence_embedding_dimension()


# Example usage
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    # Generate embeddings
    texts = [
        "Ajatuskumppani on suomalainen tekoäly.",
        "ThoughtMate is a Finnish AI platform."
    ]
    
    embeddings = generator.generate(texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {generator.embedding_dim}")
    
    # Calculate similarity
    similarity = generator.similarity(texts[0], texts[1])
    print(f"Similarity: {similarity:.4f}")

