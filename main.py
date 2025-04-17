# This is required sometimes for Linux OS
import sys
import platform
if platform.system() == "Linux":
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from utils.tools import select_app_option

if __name__ == "__main__":

    print("\n=== Dr.X Document Analysis Suite ===")

    option = select_app_option()
        
    # RAG
    if option == 0:
        from modules.rag_qa import start_rag
        start_rag()

    # Translator
    elif option == 1:
        from modules.translator import start_translator
        start_translator(target_lang="english")

    # Translator (Arabic)
    elif option == 2:
        from modules.translator import start_translator
        start_translator(target_lang="arabic")

    # Summarizer
    elif option == 3:
        from modules.summarizer import start_summarizer
        start_summarizer()

    # Evaluation
    elif option == 4:
        from modules.evaluation import start_evaluation
        start_evaluation()

    else:
        print("\nInvalid option. Please run again and select 1-5.")
