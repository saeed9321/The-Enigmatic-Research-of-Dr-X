from typing import List, Any
import os
from utils.logger import Logger
from config import get_config
import inquirer
import unicodedata
import re

logger = Logger()
config = get_config()


def normalize_text(text: str) -> str:
    """
    Text normalization for safe prompt use.
    - Lowercase
    - Remove quotes and risky characters
    - Remove punctuation
    - Collapse multiple spaces
    - Remove accents/diacritics
    - Strip surrounding whitespace
    """

    # Lowercase the text
    text = text.lower()

    # Remove smart quotes and dangerous characters
    text = text.replace('"', '').replace("'", "")
    text = text.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    text = text.replace("`", "").replace("´", "")

    # Remove triple quotes and anything that could break a prompt string
    text = text.replace('"""', '').replace("'''", "")

    # Remove punctuation using regex (except basic alphanumerics and spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize unicode (remove accents/diacritics)
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # Remove multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def _format_table(table_data: List[List[Any]]) -> str:
    """Convert table data into natural language sentences for better semantic understanding"""
    if not table_data or len(table_data) < 2:
        return ""
        
    headers = [str(cell).strip() for cell in table_data[0]]
    formatted_text = []
    
    # Create semantic description of table structure
    formatted_text.append(f"The table contains {len(table_data)-1} records with the following attributes: {', '.join(headers)}.")
    
    # Process each row as a complete sentence
    for row in table_data[1:]:
        if not any(cell for cell in row):  # Skip empty rows
            continue
            
        # Create a sentence for each row
        row_parts = []
        for header, cell in zip(headers, row):
            cell_text = str(cell).strip() if cell is not None else "NA"
            row_parts.append(f"{header} is {cell_text}")
        
        formatted_text.append("The record shows that " + ", ".join(row_parts) + ".")
    
    return "\n\n".join(formatted_text)

def select_app_option() -> int:
    """Let user select an app option using inquirer."""

    main_app_options = [
                         "1. Starting RAG QA",
                         "2. Starting Translation - English",
                         "3. Starting Translation - Arabic",
                         "4. Starting Main Ideas extraction",
                         "5. Starting Summary evaluation"
                     ]
        
    questions = [
        inquirer.List('option',
                     message="Select an option",
                     choices=main_app_options,
                    )
    ]
    answers = inquirer.prompt(questions)
    selected = answers['option']
    selected_index = main_app_options.index(selected)
    logger.info(f"Selected option: {selected}")
    return selected_index

def list_files_in_directory(directory: str) -> List[str]:
    """List all files in the specified directory."""
    try:
        files = []
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                files.append(file)
        return files
    except Exception as e:
        logger.error(f"Error listing files in directory {directory}: {str(e)}")
        return []

def select_file(files: List[str]) -> str:
    """Let user select a file using inquirer."""
    if not files:
        logger.error("No files available for selection")
        return None
        
    questions = [
        inquirer.List('file',
                     message="Select a file",
                     choices=files,
                    )
    ]
    answers = inquirer.prompt(questions)
    selected = answers['file']
    logger.info(f"Selected file: {selected}")
    return selected

def select_summarization_method() -> str:
    """Let user select a summarization method."""
    questions = [
        inquirer.List('method',
                     message="Select a summarization method",
                     choices=[
                         ('Map-Reduce (Best for large documents)', 'map_reduce'),
                         ('Refine (Best for maintaining context)', 'refine'),
                         ('Rerank (Best for extracting key information)', 'rerank')
                     ],
                    )
    ]
    answers = inquirer.prompt(questions)
    selected = answers['method']
    logger.info(f"Selected summarization method: {selected}")
    return selected

def select_prompt_strategy() -> str:
    """Let user select a prompt strategy."""
    questions = [
        inquirer.List('prompt_strategy',
                     message="Select a prompt strategy",
                     choices=config["prompt_strategies"].keys(),
                    )
    ]
    answers = inquirer.prompt(questions)
    selected = answers['prompt_strategy']
    logger.info(f"Selected prompt strategy: {selected}")
    return selected
  
def select_chunking_strategy() -> str:
    """Let user select a chunking strategy."""
    questions = [
        inquirer.List('chunking_strategy',
                     message="Select a chunking strategy",
                     choices=["recursive", "agented"],
                    )
    ]
    answers = inquirer.prompt(questions)
    selected = answers['chunking_strategy']
    logger.info(f"Selected chunking strategy: {selected}")
    return selected
