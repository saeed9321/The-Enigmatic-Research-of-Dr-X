import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Literal, Optional
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
import time
from tqdm import tqdm
from config import get_config
from utils.logger import Logger
from utils.tools import normalize_text
tokenizer = tiktoken.get_encoding('cl100k_base')

class Chunker:
    def __init__(self, strategy: Optional[Literal['recursive', 'agented']] = None, chunk_overlap: Optional[int] = 0):
        self.config = get_config()
        # Load default configuration
        self.model = self.config["llm"].get("default_model")
        self.chunk_strategy = strategy or self.config["chunking"].get("chunk_strategy")
        self.max_tokens = self.config["chunking"].get("max_tokens_per_chunk")
        self.chunk_overlap = chunk_overlap or self.config["chunking"].get("chunk_overlap")

        self.logger = Logger(log_file=self.config["chunking"]["log_file"])
        
        # Only use the llm if strategy used is 'agented'
        self.llm = OllamaLLM(model=self.model) 

    def chunk_docs(self, documents: List[Document]) -> List[Dict]:
        start_time = time.time()
        chunks = []

        if self.chunk_strategy == 'recursive':
            chunks = self._recursive_chunking(documents)
        elif self.chunk_strategy == 'agented':
            chunks = self._agented_chunking(documents)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunk_strategy}")
        
        self.logger.metrics(
            "Document Chunking", 
            start_time, 
            time.time(), 
            {
                "strategy": self.chunk_strategy,
                "chunks_created": len(chunks)
            }
        )
        return chunks

    def _recursive_chunking(self, documents: List[Document]) -> List[Dict]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.max_tokens,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = []
        chunk_id = 1
        
        for doc in documents:
            is_structured_row = doc.metadata.get('document_type') in ['csv_row', 'excel_row']
            
            if is_structured_row:
                token_count = len(tokenizer.encode(doc.page_content))
                metadata = {
                    "source": doc.metadata['source'],
                    "chunk_id": chunk_id,
                    "row_number": doc.metadata.get('row_number'),
                    "document_type": doc.metadata['document_type'],
                    "chunk_size": token_count
                }
                chunks.append({"page_content": doc.page_content, "metadata": metadata})
                chunk_id += 1
            else:
                section_chunks = text_splitter.split_documents([doc])
                section_chunks = [chunk for chunk in section_chunks 
                                if chunk.page_content is not None and len(chunk.page_content.split()) > 3]
                
                for chunk in section_chunks:
                    token_count = len(tokenizer.encode(chunk.page_content))
                    metadata = {
                        "source": chunk.metadata['source'],
                        "chunk_id": chunk_id,
                        "page_number": chunk.metadata.get('page_number', 1),
                        "chunk_size": token_count
                    }
                    chunks.append({"page_content": chunk.page_content, "metadata": metadata})
                    chunk_id += 1
        
        return chunks

    def _process_structured_data(self, doc: Document, chunk_id: int) -> Dict:
        """Process structured data like CSV/Excel rows."""
        token_count = len(tokenizer.encode(doc.page_content))
        return {
            "page_content": doc.page_content,
            "metadata": {
                "source": doc.metadata['source'],
                "chunk_id": chunk_id,
                "row_number": doc.metadata.get('row_number'),
                "document_type": doc.metadata['document_type'],
                "chunk_size": token_count
            }
        }

    def _create_chunk_metadata(self, doc: Document, chunk_id: int, token_count: int) -> Dict:
        """Create metadata for a chunk."""
        return {
            "source": doc.metadata['source'],
            "chunk_id": chunk_id,
            "page_number": doc.metadata.get('page_number', 1),
            "chunk_size": token_count,
            "chunking_method": "ai"
        }

    def _get_ai_response(self, doc: Document) -> List[Dict]:
        """Get AI-generated chunks from the document."""
        prompt = """
        Analyze this document and split it into meaningful chunks that preserve semantic coherence.
        Each chunk should be self-contained and make sense on its own.
        Return the chunks as a JSON formatted list where each item has 'text' field only.
        
        Document content:
        {content}
        
        JSON Response format:
        [
            {{"text": "chunk 1 content"}},
            {{"text": "chunk 2 content"}},
            ...
        ]
        """

        normalized_chunk_content = normalize_text(doc.page_content)
        prompt = prompt.format(content=normalized_chunk_content)
        
        chain = self.llm | JsonOutputParser()
        response = chain.invoke(prompt)
 
        return response

    def _agented_chunking(self, documents: List[Document]) -> List[Dict]:
        """Chunk documents using AI-based semantic chunking with fallback to recursive chunking."""
        chunks = []
        chunk_id = 1
        
        for doc in tqdm(documents, desc="Agented Chunking"):
            if not doc.page_content or len(doc.page_content.strip()) == 0:
                continue
                
            if doc.metadata.get('document_type') in ['csv_row', 'excel_row']:
                chunks.append(self._process_structured_data(doc, chunk_id))
                chunk_id += 1
                continue
            
            try:
                ai_chunks = self._get_ai_response(doc)
                
                for chunk in ai_chunks:
                    if not chunk.get('text', '').strip():
                        continue
                    
                    token_count = len(tokenizer.encode(chunk['text']))
                    if token_count < 10:
                        continue
                    
                    chunks.append({
                        "page_content": chunk['text'],
                        "metadata": self._create_chunk_metadata(doc, chunk_id, token_count)
                    })
                    chunk_id += 1
                    
            except Exception as e:
                self.logger.error(f"AI chunking failed: {str(e)}")
                self.logger.warning(f"Fallback from agented to recursive chunking due to: {str(e)}")
                return self._recursive_chunking(documents)

        if not chunks:
            self.logger.warning("AI chunking produced no chunks, falling back to recursive chunking")
            return self._recursive_chunking(documents)
            
        return chunks
