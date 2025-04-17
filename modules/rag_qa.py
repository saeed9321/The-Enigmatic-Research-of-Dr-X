import os
import time
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from tiktoken import get_encoding

# Import RAG modules
from modules.file_loader import EnhancedDocumentLoader
from modules.chunker import Chunker
from modules.embedder import Embedder
from modules.vector_db import VectorDB
from config import get_config
from utils.logger import Logger
from utils.tools import select_chunking_strategy


class RAGPipeline:
    """A RAG (Retrieval-Augmented Generation) system for answering questions"""
    
    def __init__(self, config=None):
        """Initialize the RAG Pipeline with configuration."""
        self.config = config if config is not None else get_config()
        self.logger = Logger(self.config["rag"]["log_file"])
        self.logger.info("Initializing RAG Pipeline")
        
        # Initialize tokenizer and LLM
        self.encoder = get_encoding("cl100k_base")
        self.model_name = self.config["llm"].get("default_model")
        self.llm = ChatOllama(
            model=self.model_name, 
            num_ctx=self.config["llm"].get("num_ctx"),
            num_predict=self.config["llm"].get("max_output_tokens"),
            temperature=self.config["llm"].get("temperature")
        )
        self.logger.success(f"Initialized LLM: {self.model_name}")
        
        # Initialize vector database retriever
        self.vector_db_retriever = None
        self.reset_token_metrics()

    def reset_token_metrics(self):
        """Reset token metrics for a new session."""
        self.token_metrics = {
            "tokens_sent": {"user_questions": 0, "system_prompt": 0, "total": 0},
            "tokens_received": {"ai_responses": 0, "total": 0},
            "tokens/second": 0,
            "memory_enabled": False
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoder.encode(text))
    
    def process_documents_pipeline(self, files_path: List[str]) -> Any:
        """Process documents through the RAG pipeline."""
        try:
            # Process documents
            loader = EnhancedDocumentLoader()
            processed_docs = loader.load_multiple_documents(files_path=files_path)
            
            # Chunk documents
            # Let user select chunking strategy
            selected_chunking_strategy = select_chunking_strategy()
            if not selected_chunking_strategy:
                self.logger.error("No chunking strategy was selected.")
                return
            chunker = Chunker(strategy=selected_chunking_strategy)
            chunked_docs = chunker.chunk_docs(documents=processed_docs)
            
            # Generate embeddings
            embedding_model = self.config['embedding']['model_name']
            embedder = Embedder(embedding_model=embedding_model)
            embedded_docs = embedder.generate_embeddings(chunked_docs)
            
            # Create vector database
            vector_db_retriever = VectorDB(chunks=embedded_docs)           
            
            return vector_db_retriever
            
        except Exception as e:
            self.logger.error(f"Error in document processing pipeline: {str(e)}")
            raise
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        return "\n".join([
            f"\n{doc['page_content']}\nMetadata: {doc['metadata']}"
            for doc in results
        ])

    def _update_token_metrics(self, text: str, metric_key: str) -> int:
        """Update token metrics for given text and return token count."""
        tokens = self.count_tokens(text)
        self.token_metrics["tokens_sent"][metric_key] = tokens
        self.token_metrics["tokens_sent"]["total"] += tokens
        return tokens

    def _build_chat_messages(self, system_content: str, chat_history: List[Dict[str, str]]) -> tuple[List, List[Dict[str, str]]]:
        """Build message list including chat history if available."""
        messages = [SystemMessage(content=system_content)]
        
        if len(chat_history) == 0:
            return messages, []
            
        max_history = self.config["rag"].get("max_history_messages", 2)
        if len(chat_history) > max_history * 2:
            chat_history = chat_history[-(max_history * 2):]
            self.logger.debug(f"Trimmed chat history to last {max_history} exchanges")
        
        self.logger.debug(f"Added chat history to messages")
        messages.extend([SystemMessage(content="The below is the chat history:")])
        messages.extend([
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in chat_history
        ])
        return messages, chat_history

    def generate_answer(
        self,
        question: str,
        retriever: Any,
        chat_history: List[Dict[str, str]] = [],
        n_results: int = 5
    ) -> List[Dict[str, str]]:
        """Generate an answer using RAG."""
        try:
            # Retrieve and process documents
            results = retriever.query(query_text=question, n_results=n_results)
            self.logger.debug(f"Retrieved {len(results)} documents for query")
            self.logger._log_to_file(results)
            
            # Build context and system message
            context = self._build_context(results)
            system_content = self.config["system_prompt_template"].format(context=context, question=question)
            self._update_token_metrics(system_content, "system_prompt")
            
            # Generate response and get potentially trimmed history
            messages, trimmed_history = self._build_chat_messages(system_content, chat_history)
            
            # Add the current question to messages
            messages.append(HumanMessage(content=question))
            
            # Make LLM call
            start_time = time.time()
            response = self.llm.invoke(messages)
            
            # Update metrics
            response_tokens = self.count_tokens(response.content)
            elapsed_time = time.time() - start_time
            self.token_metrics["tokens/second"] = response_tokens / elapsed_time
            self.token_metrics["tokens_received"]["ai_responses"] = response_tokens
            self.token_metrics["tokens_received"]["total"] += response_tokens
            
            # Update chat history with new messages
            trimmed_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response.content}
            ])
            
            return trimmed_history
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return f"I'm sorry, I encountered an error: {str(e)}", chat_history
    
    def run_chat_session(self):
        """Run an interactive chat session with the user."""
        if not self.vector_db_retriever:
            self.logger.error("Vector DB retriever not initialized. Please process documents first.")
            return
        
        # Get settings from config
        use_history = self.config["rag"].get("use_history", True)
        n_results = self.config["rag"].get("n_results", 5)
        title = self.config["ui"].get("welcome_title", "Dr. X's Research Assistant")
        subtitle = self.config["ui"].get("welcome_subtitle", "Ask questions about Dr. X's research documents")
        exit_commands = self.config["ui"].get("exit_commands", ["exit", "quit", "bye"])
            
        # Initialize chat history and display welcome
        chat_history = []
        self.logger.print_welcome(title, subtitle)
        
        # Set memory status in metrics
        self.token_metrics["memory_enabled"] = use_history
        
        # Chat loop
        while True:
            # Get user input
            question = input("🧑 User: ").strip()
            
            if question.lower() in exit_commands:
                self.logger.info("Thank you for using Dr. X's Research Assistant. Goodbye!")
                break
                
            if not question:
                continue
            
            # Update metrics and generate answer
            question_tokens = self.count_tokens(question)
            self.token_metrics["tokens_sent"]["user_questions"] += question_tokens
            
            with self.logger.progress() as progress:
                progress.add_task("Thinking...", total=None)
                start_time = time.time()
                updated_chat_history = self.generate_answer(
                    question, 
                    self.vector_db_retriever, 
                    chat_history if use_history else [], 
                    n_results
                )

            # Update chat history
            chat_history = updated_chat_history
            
            # Display last two messages using the updated chat history
            for msg in updated_chat_history[-2:]:
                self.logger.print_chat_message(msg["role"], msg["content"])
            self.logger.print_separator()
            
            # Log metrics
            self.logger.metrics("Question Answering", start_time, time.time(), self.token_metrics)


def start_rag():
    """Entry point for the RAG system."""
    try:
        # Get configuration
        config = get_config()
        
        # Get config values
        docs_dir = config["document"]["documents_dir"]
        
        # Initialize logger and get file paths
        logger = Logger(config["rag"]["log_file"])
        start_time = time.time()
        logger.metrics("Started Pipeline", time.time(), time.time())
        
        if not os.path.exists(docs_dir):
            logger.error(f"Documents directory does not exist: {docs_dir}")
            return
                
        files_path = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir)]
        if not files_path:
            logger.error(f"No files found in documents directory: {docs_dir}")
            return
                
        logger.info(f"Found {len(files_path)} files in {docs_dir}")
        
        # Initialize RAG pipeline with config
        rag = RAGPipeline(config)
        
        # Check if vector DB exists
        vector_db_retriever = VectorDB()
        
        if vector_db_retriever.collection.count() == 0:
            logger.info("Processing new documents...")
            rag.vector_db_retriever = rag.process_documents_pipeline(files_path)
        else:
            logger.info(f"Using existing Vector DB with {vector_db_retriever.collection.count()} documents")
            rag.vector_db_retriever = vector_db_retriever
        
        # Log overall pipeline execution
        logger.metrics("Overall Pipeline Execution", start_time, time.time())

        # Run interactive chat session
        rag.run_chat_session()
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise
