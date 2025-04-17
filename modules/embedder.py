from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils.logger import Logger
import time
from config import get_config
from tiktoken import get_encoding
from typing import Optional


class Embedder:
    def __init__(self, embedding_model: Optional[str] = None):
        self.config = get_config()
        self.logger = Logger(log_file=self.config["embedding"]["log_file"])
        self.embeddingModel = SentenceTransformer(embedding_model or self.config["embedding"]["model_name"], trust_remote_code=True)
        self.encoder = get_encoding("cl100k_base")

    def generate_embeddings(self, chunks):
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of dictionaries containing at least a 'text' key
            
        Returns:
            The same list with 'embedding' added to each chunk
        """

        start_time = time.time()
        self.logger.info("Generating embeddings")
        
        texts = [chunk['page_content'] for chunk in chunks]
        embeddings = []
        for text in tqdm(texts, desc="Generating Embeddings"):
            # Calculate tokens per second for this text chunk
            tokens = len(self.encoder.encode(text))
            elapsed_time = time.time() - start_time
            tokens_per_second = tokens / elapsed_time if elapsed_time > 0 else 0
            self.logger.info(f"Processing speed: {tokens_per_second:.2f} tokens/second")
            embedding = self.embeddingModel.encode([text])[0]
            embeddings.append(embedding)

        for i, embedding in enumerate(embeddings):
            chunks[i]['embedding'] = embedding.tolist()

        self.logger.metrics(
                "Embedding Generation", 
                start_time, 
                time.time(), 
                {"embeddings_created": len(chunks)}
            )
        return chunks


