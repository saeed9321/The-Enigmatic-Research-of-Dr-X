import os
from langchain_ollama import OllamaLLM
from tqdm import tqdm
from modules.file_loader import EnhancedDocumentLoader
from modules.chunker import Chunker
import time
from utils.logger import Logger
from config import get_config
from utils.tools import select_file, list_files_in_directory
from typing import Optional
from tiktoken import get_encoding

class Translator:
    def __init__(self, model: str, target_lang: str):
        self.target_lang = target_lang
        self.config = get_config()
        self.logger = Logger(log_file=self.config["translation"]["log_file"])
        self.model = model or self.config["translation"]["default_model"]
        self.encoder = get_encoding("cl100k_base")

        # Create LLM Call
        self.llm = OllamaLLM(model=self.model)
        self.prompt = """
            Translate the following text to {language}. 
            Keep all formatting, headings, and bullet points exactly as in the original.
            Only output the translated text, no other text or comments. if you are not sure then return the original text only.

            Text:
            {text}
        """

    def translate_text(self, text: str):

        # Calculate the number of tokens in the text
        num_tokens = len(self.encoder.encode(text))

        complete_prompt = self.prompt.format(text=text, language=self.target_lang)
        
        start_time = time.time()
        translated_text = self.llm.invoke(complete_prompt)
        end_time = time.time()
        
        time_taken = end_time - start_time
        self.logger.info(f"Translating {len(text.split())} words completed in {time_taken:.2f} seconds. (model: {self.model}, tokens/second: {num_tokens / time_taken:.2f})")

        return translated_text

def start_translator(model: Optional[str] = None, files_path: str = "", target_lang: str = ""):
    """
    Main function to start the translation process with a CLI interface.
    """
    config = get_config()
    logger = Logger(log_file=config["translation"]["log_file"])
    model = model or config["translation"]["default_model"]
    files_path = files_path or files_path or config["document"]["documents_dir"]

    # Define directories
    input_dir = files_path
    output_dir = "translation_output"
    os.makedirs(output_dir, exist_ok=True)

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

    # Get selected file path
    input_file_path = os.path.join(input_dir, selected_file)
    output_file_path = os.path.join(output_dir, f"translation_{selected_file}_{target_lang}.txt")

    # Read the content of the selected file using EnhancedDocumentLoader
    try:
        loader = EnhancedDocumentLoader()
        documents = loader.load_document(input_file_path)
    except Exception as e:
        print(f"Error loading document '{selected_file}': {e}")
        return

    # Initialize the translator
    translator = Translator(model=model, target_lang=target_lang)

    # Chunk the document
    chunker = Chunker(strategy="recursive")
    chunks = chunker.chunk_docs(documents)

    # Translate each chunk and stream output
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for chunk in tqdm(chunks, desc="Translating"):
            translated_chunk = translator.translate_text(chunk['page_content'])
            output_file.write(translated_chunk)
            output_file.flush()

    logger.success(f"Translation completed. Output saved to {output_file_path}")

