from modules.file_loader import EnhancedDocumentLoader
from modules.chunker import Chunker
from modules.embedder import Embedder
from modules.evaluation import SummaryEvaluator
from utils.tools import list_files_in_directory,select_chunking_strategy, select_file, select_summarization_method, select_prompt_strategy
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import List, Dict, Any, Optional
from utils.logger import Logger
from config import get_config

class Summarizer:
    """
    A class that handles document summarization using clustering and LLM-based summarization.
    
    Attributes:
        model (str): Name of the LLM model to use for summarization
        n_clusters (int): Number of clusters to group similar document sections
    """
    
    def __init__(self, model: Optional[str] = None, n_clusters: Optional[int] = None, chunking_strategy: Optional[str] = None):
        self.config = get_config()
        self.logger = Logger(log_file=self.config["summarizer"]["log_file"])
        self.n_clusters = n_clusters or self.config["clustering"]["n_clusters"]
        self.chunking_strategy = chunking_strategy or self.config["chunking"]["chunk_strategy"]
        start_time = time.time()
        
        self.model = model or self.config["llm"]["default_model"]
        
        try:
            self.llm = ChatOllama(model=self.model)
            self.loader = EnhancedDocumentLoader()
            self.evaluator = SummaryEvaluator()
            self.chunker = Chunker(strategy=self.chunking_strategy)
            self.embedder = Embedder()

            end_time = time.time()
            self.logger.metrics("Summarizer Initialization", start_time, end_time, {
                "model": self.model,
            })
        except Exception as e:
            self.logger.error(f"Failed to initialize Summarizer: {str(e)}")
            raise

    def visualize_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray, documents: List[Dict[str, Any]]):
        """
        Visualize the clusters using t-SNE.
        
        Args:
            embeddings: Array of document embeddings
            cluster_labels: Array of cluster labels
            documents: List of document dictionaries containing the actual content
        """
        try:
            n_samples = embeddings.shape[0]
            
            # Calculate appropriate perplexity
            perplexity = min(n_samples - 1, 30)
            
            if n_samples < 4:
                self.logger.warning("Not enough samples for t-SNE visualization (minimum 4 required)")
                return
                
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)

            # Create figure for scatter plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=cluster_labels, cmap='viridis')
            plt.title(f'Cluster Visualization using t-SNE (perplexity={perplexity})')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.colorbar(scatter, label='Cluster Label')
            
            # Save the visualization
            os.makedirs('logs/visualizations', exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f'logs/visualizations/cluster_visualization_{timestamp}.png'
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save cluster information to a separate text file
            text_filepath = f'logs/visualizations/cluster_contents_{timestamp}.txt'
            with open(text_filepath, 'w') as f:
                f.write("Cluster Contents:\n\n")
                cluster_contents = {}
                for i, label in enumerate(cluster_labels):
                    if label not in cluster_contents:
                        cluster_contents[label] = []
                    cluster_contents[label].append(documents[i]['page_content'])
                
                for cluster_id, contents in cluster_contents.items():
                    f.write(f"Cluster {cluster_id}:\n")
                    for text in contents:
                        f.write(f"• {' '.join(text.split())}\n")
                    f.write(f"Total documents in cluster: {len(contents)}\n\n")
            
            self.logger.info(f"Cluster visualization saved to {filepath}")
            self.logger.info(f"Cluster contents saved to {text_filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to visualize clusters: {str(e)}")

    def cluster_documents(self, documents: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster documents based on their embeddings using KMeans.
        
        Args:
            documents: List of document dictionaries containing embeddings
            
        Returns:
            Dictionary mapping cluster IDs to lists of documents
            
        Raises:
            ValueError: If documents list is empty or embeddings are missing
        """
        if not documents:
            raise ValueError("No documents provided for clustering")
                    
        try:
            # Extract embeddings
            embeddings = np.array([doc['embedding'] for doc in documents])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(self.n_clusters, len(documents)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Visualize the clusters
            self.visualize_clusters(embeddings, cluster_labels, documents)
            
            # Group documents by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(documents[i])
            
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")
            raise

    def summarize_cluster(self, cluster_id: int, documents: List[Dict[str, Any]], method: str = "map_reduce", prompt_strategy: str = "default") -> str:
        """
        Generate a summary for a cluster of documents using the specified method and prompt strategy.
        
        Args:
            documents: List of document dictionaries
            method: Summarization method ('map_reduce', 'refine', or 'rerank')
            prompt_strategy: The prompt strategy to use ('default', 'analytical', or 'extractive')
        """
        start_time = time.time()
        
        # Convert to langchain Document format
        docs = [Document(page_content=doc['page_content'], metadata=doc.get('metadata', {})) 
                for doc in documents]
        
        # Get prompt templates for the selected strategy
        prompts = self.config["prompt_strategies"][prompt_strategy]
        map_prompt = PromptTemplate(template=prompts["map_template"], input_variables=["text"])
        combine_prompt = PromptTemplate(template=prompts["combine_template"], input_variables=["text"])
        
        chain_start = time.time()
        
        if method == "map_reduce":
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt
            )
            summary = chain.invoke(docs)
        elif method == "refine":
            chain = load_summarize_chain(
                self.llm,
                chain_type="refine",
                question_prompt=map_prompt,
                refine_prompt=combine_prompt
            )
            summary = chain.invoke(docs)
        elif method == "rerank":
            chain = load_summarize_chain(
                self.llm,
                chain_type="stuff",
                prompt=map_prompt
            )
            summaries = []
            for i, doc in enumerate(docs):
                summary = chain.invoke([doc])
                score = {
                    "length": len(summary['output_text']),
                    "info_density": len(set(summary['output_text'].split())) / len(summary['output_text'].split()),
                    "chunk_index": i
                }
                total_score = score["length"] * score["info_density"]
                
                summaries.append({
                    "text": summary['output_text'], 
                    "score": total_score,
                    "metrics": score
                })
                
            summaries.sort(key=lambda x: x["score"], reverse=True)
            summary = {"output_text": "\n".join([s["text"] for s in summaries[:3]])}
        else:
            raise ValueError(f"Unsupported summarization method: {method}")
        
        chain_end = time.time()
        
        end_time = time.time()
        self.logger.metrics(f"Create Main Idea {cluster_id}", start_time, end_time, {
            "method": method,
            "prompt_strategy": prompt_strategy,
            "n_documents": len(documents),
            "original_length_words": sum(len(_doc['page_content'].split()) for _doc in documents),
            "summary_length_words": len(summary['output_text'].split()),
            "chain_processing_time": chain_end - chain_start,
            "tokens/second": len(summary['output_text'].split()) / (end_time - start_time)
        })
        
        return summary['output_text']

    def summarize_cluster_summaries(self, cluster_summaries: List[str]) -> str:
        """
        Summarize a list of cluster summaries into a final summary.
        
        Args:
            cluster_summaries: List of cluster summaries
            
        Returns:
            Final summary of the cluster summaries
        """

        start_time = time.time()
        self.logger.info("Generating final summary from cluster summaries...")
        
        # Combine all cluster summaries into a single text
        combined_text = "\n\n".join([text for text in cluster_summaries])
        
        # Create the prompt for final summarization
        prompt_template = """
        You are an expert summarizer. Your task is to create a comprehensive, coherent summary 
        of the following text, which consists of multiple cluster summaries. 
        
        Focus on maintaining the key points while creating a unified narrative.
        
        TEXT TO SUMMARIZE:
        {text}
        
        COMPREHENSIVE SUMMARY:
        """
        
        prompt = prompt_template.format(text=combined_text)
        
        # Generate the final summary
        llm_answer = self.llm.invoke(prompt)
        final_summary = llm_answer.content
        
        end_time = time.time()
        self.logger.metrics("Final Summarization", start_time, end_time, {
            "input_clusters": len(cluster_summaries),
            "input_length_words": len(combined_text.split()),
            "output_length_words": len(final_summary.split()),
            "processing_time": end_time - start_time
        })
        
        return final_summary

    def summarize_chunks(self, chunks: List[Dict[str, Any]], method: str = "map_reduce", prompt_strategy: str = "default") -> str:
        """
        Summarize a list of chunks using the specified method and prompt strategy.
        
        Args:
            chunks: List of chunk dictionaries
        """
        start_time = time.time()
        
        # Convert to langchain Document format
        docs = [Document(page_content=doc['page_content'], metadata=doc.get('metadata', {})) 
                for doc in chunks]
        
        # Get prompt templates for the selected strategy
        prompts = self.config["prompt_strategies"][prompt_strategy]
        map_prompt = PromptTemplate(template=prompts["map_template"], input_variables=["text"])
        combine_prompt = PromptTemplate(template=prompts["combine_template"], input_variables=["text"])
        
        chain_start = time.time()
        
        if method == "map_reduce":
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt
            )
            summary = chain.invoke(docs)
        elif method == "refine":
            chain = load_summarize_chain(
                self.llm,
                chain_type="refine",
                question_prompt=map_prompt,
                refine_prompt=combine_prompt
            )
            summary = chain.invoke(docs)
        elif method == "rerank":
            chain = load_summarize_chain(
                self.llm,
                chain_type="stuff",
                prompt=map_prompt
            )
            summaries = []
            for i, doc in enumerate(docs):
                summary = chain.invoke([doc])
                score = {
                    "length": len(summary['output_text']),
                    "info_density": len(set(summary['output_text'].split())) / len(summary['output_text'].split()),
                    "chunk_index": i
                }
                total_score = score["length"] * score["info_density"]
                
                summaries.append({
                    "text": summary['output_text'], 
                    "score": total_score,
                    "metrics": score
                })
                
            summaries.sort(key=lambda x: x["score"], reverse=True)
            summary = {"output_text": "\n".join([s["text"] for s in summaries[:3]])}
        else:
            raise ValueError(f"Unsupported summarization method: {method}")
        
        chain_end = time.time()
        
        end_time = time.time()
        self.logger.metrics("Chunk Summarization", start_time, end_time, {
            "method": method,
            "prompt_strategy": prompt_strategy,
            "n_chunks": len(chunks),
            "original_length_words": sum(len(_doc['page_content'].split()) for _doc in chunks),
            "summary_length_words": len(summary['output_text'].split()),
            "chain_processing_time": chain_end - chain_start,
            "tokens/second": len(summary['output_text'].split()) / (end_time - start_time)
        })

        return summary['output_text']
        
    def generate_summary(self, file_path: str, with_clustering: bool = False, method: str = "map_reduce", prompt_strategy: str = "default", chunking_strategy: str = "recursive") -> str:
        """
        Generate a comprehensive summary of a file using clustering and LLM summarization or without clustering.
        
        Args:
            file_path: Path to the file to summarize
            method: Summarization method to use
        """
        start_time = time.time()
        
        try:
            # Load and process the document
            documents = self.loader.load_document(file_path)

            # Chunk the documents
            chunks = self.chunker.chunk_docs(documents)

            summary_duration = 0

            if not with_clustering:
                final_summary = self.summarize_chunks(chunks)
                summary_duration = time.time() - start_time
            else:
                # Generate embeddings
                docs_with_embeddings = self.embedder.generate_embeddings(chunks)
                
                # Cluster the documents
                cluster_start = time.time()
                self.logger.info("Clustering documents...")
                clusters = self.cluster_documents(docs_with_embeddings)
                cluster_end = time.time()
                cluster_duration = cluster_end - cluster_start
                self.logger.metrics("Document Clustering: Extract Main Idea", cluster_start, cluster_end, {
                "n_clusters": len(clusters),
                "avg_docs_per_cluster": len(docs_with_embeddings) / len(clusters),
                "duration_seconds": cluster_duration
            })
                self.logger.info(f"Document clustering completed in {cluster_duration:.2f} seconds")

                # Generate summaries for each cluster
                summary_start = time.time()
                self.logger.info("Generating summaries for each cluster...")
                cluster_summaries = []
                with self.logger.progress("Processing clusters") as progress:
                    task = progress.add_task("Summarizing", total=len(clusters))
                    for cluster_id, cluster_docs in clusters.items():
                        summary = self.summarize_cluster(
                            cluster_id,
                            cluster_docs,
                            method=method,
                            prompt_strategy=prompt_strategy
                        )
                        cluster_summaries.append(summary)
                        progress.advance(task)
                summary_end = time.time()
                summary_duration = summary_end - summary_start
            
                # Summarize the cluster summaries into a final summary
                final_summary = self.summarize_cluster_summaries(cluster_summaries)

            end_time = time.time()
            total_duration = end_time - start_time
        
            # Log final metrics
            self.logger.metrics("Complete Document Summary", start_time, end_time, {
                "file": file_path,
                "method": method,
                "prompt_strategy": prompt_strategy,
                "chunking_strategy": chunking_strategy,
                "n_clusters": len(clusters) if with_clustering else 0,
                "summarization_time": summary_duration,
                "total_time": total_duration
            })

            self.logger.info(f"Summary: {final_summary}")
            
            # Get Human Generated Summary file for this file
            file_name = os.path.basename(file_path)
            base_dir = self.config["base_dir"]
            reference_summary_path = f"{base_dir}/summary_files_organic/{file_name}.txt"
            
            # Read the reference summary from the file
            with open(reference_summary_path, 'r') as f:
                reference_summary = f.read()

            # Evaluate the summary
            self.evaluator.evaluate_summary(generated_summary=final_summary, reference_summary=reference_summary, method=method, prompt_strategy=prompt_strategy)
        
            return final_summary
            
        except Exception as e:
            self.logger.error(f"An error occurred while generating the summary: {str(e)}")
            raise


def start_summarizer():
    """
    Main function to start the summarization process with a CLI interface.
    Handles user interaction and error cases gracefully.
    """
    # Use config values if not provided
    config = get_config()
    files_path = config["document"]["documents_dir"]
    model = config["llm"]["default_model"]
    n_clusters = config["clustering"]["n_clusters"]
    with_clustering = config["summarizer"]["with_clustering"]
    
    logger = Logger(log_file=config["summarizer"]["log_file"])
    logger.print_welcome(
        config["ui"]["welcome_title"],
        "Document Summarization System"
    )
    
    # Validate directory exists
    if not os.path.exists(files_path):
        logger.error(f"The directory '{files_path}' does not exist.")
        return
    
    # List all files
    files = list_files_in_directory(files_path)
    if not files:
        logger.error("No files found in the directory.")
        return

    # Let user select a file
    selected_file = select_file(files)
    if not selected_file:
        logger.error("No file was selected.")
        return
        
    # Let user select summarization method
    selected_method = select_summarization_method()
    if not selected_method:
        logger.error("No summarization method was selected.")
        return
        
    # Let user select prompt strategy
    selected_prompt_strategy = select_prompt_strategy()
    if not selected_prompt_strategy:
        logger.error("No prompt strategy was selected.")
        return
    
    # Let user select chunking strategy
    selected_chunking_strategy = select_chunking_strategy()
    if not selected_chunking_strategy:
        logger.error("No chunking strategy was selected.")
        return

    file_path = os.path.join(files_path, selected_file)

    logger.info(f"Clustering with KMeans: {with_clustering}")
    
    try:
        # Initialize summarizer and generate summary
        summarizer = Summarizer(model=model, n_clusters=n_clusters, chunking_strategy=selected_chunking_strategy)
        summarizer.generate_summary(file_path, with_clustering=with_clustering, method=selected_method, prompt_strategy=selected_prompt_strategy)

    except Exception as e:
        logger.error(f"An error occurred while processing: {str(e)}")
        logger.info("Please check the log file for more details.")

