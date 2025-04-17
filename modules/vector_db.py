import chromadb
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import os
import uuid
from config import get_config
import time
from utils.logger import Logger
class VectorDB:
    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None, persist_dir: str = "./chroma_store", collection_name: str = "documents"):
        """
        Initialize the vector database.
        
        Args:
            chunks: List of dictionaries with 'text', 'embedding', and optional metadata
            persist_dir: Directory to persist the database
            collection_name: Name of the collection to store documents
        """
        self.persist_directory = persist_dir
        os.makedirs(self.persist_directory, exist_ok=True)

        # loade default config
        self.config = get_config()
        
        # Use the same Nomic model as in embedder.py
        embedding_model = self.config['embedding']['model_name']
        self.model = SentenceTransformer(embedding_model, trust_remote_code=True)
        
        # Create the ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        start_time = time.time()
        self.logger = Logger(log_file=self.config["vector_db"]["log_file"])
        self.logger.info("Storing in vector DB")

        # Check if collection exists and has chunks
        try:
            existing_collection = self.client.get_collection(name=collection_name)
            count = existing_collection.count()
            print(f"Found existing collection '{collection_name}' with {count} chunks")
            self.collection = existing_collection
        except Exception as e:
            # Collection doesn't exist or there was an error
            print(f"Creating new collection: {collection_name}")
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Using cosine similarity
            )


        if chunks and len(chunks):
            self.add_documents(chunks)

        indexed_length = len(chunks) if chunks else self.collection.count()
        self.logger.metrics(
                "Vector DB Creation", 
                start_time, 
                time.time(), 
                {"documents_indexed": indexed_length}
            )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add documents with pre-computed embeddings to the vector database.
        
        Args:
            chunks: List of dictionaries with 'text', 'embedding', and optional metadata
        """
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            if "embedding" not in chunk:
                raise ValueError("Each chunk must contain an 'embedding' key")
            if "page_content" not in chunk:
                raise ValueError("Each chunk must contain a 'page_content' key")
            
            # Create a unique ID for each document
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # Add the page_content
            documents.append(chunk["page_content"])
            
            # Add the embedding
            embeddings.append(chunk["embedding"])
            
            # Add metadata (everything except page_content and embedding)
            # Flatten nested dictionaries in metadata since Chroma expects simple types
            metadata = {}
            for k, v in chunk.items():
                if k not in ["page_content", "embedding"]:
                    # Ensure metadata values are primitive types (str, int, float, bool)
                    if isinstance(v, dict):
                        # Convert dict to string representation
                        metadata[k] = str(v)
                    elif isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
                    else:
                        # Convert any other types to string
                        metadata[k] = str(v)
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database with a text query.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return, default is 10
            
        Returns:
            List of dictionaries containing matched documents and metadata
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query_text).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "page_content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if "distances" in results else None
            })
        
        return formatted_results

