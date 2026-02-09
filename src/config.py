import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load file .env dari root directory
# Kita menggunakan Path untuk mencari file .env secara otomatis
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    """
    Centralized Configuration Management.
    Mengakses semua variabel environment dari satu tempat.
    """
    
    # --- PROJECT PATHS ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # --- LLM CONFIGURATION (Brain) ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    
    # --- CONFLUENCE CONFIGURATION (Source) ---
    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
    
    # --- QTEST CONFIGURATION (Destination) ---
    QTEST_URL = os.getenv("QTEST_URL")
    QTEST_API_TOKEN = os.getenv("QTEST_API_TOKEN")
    QTEST_PROJECT_ID = os.getenv("QTEST_PROJECT_ID")
    QTEST_PARENT_MODULE_ID = os.getenv("QTEST_PARENT_MODULE_ID")
    
    # --- VECTOR DATABASE (Knowledge Store) ---
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "vector_store"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "bri_requirements_knowledge")

    @classmethod
    def validate(cls):
        """
        Memastikan variable critical tidak kosong saat aplikasi start.
        Jika kosong, aplikasi akan error di awal (Fail Fast).
        """
        critical_vars = {
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
            "CONFLUENCE_URL": cls.CONFLUENCE_URL,
            "CONFLUENCE_API_TOKEN": cls.CONFLUENCE_API_TOKEN,
            "QTEST_URL": cls.QTEST_URL
        }

        missing = [key for key, val in critical_vars.items() if not val]
        
        if missing:
            raise ValueError(f"CRITICAL ERROR: Variabel environment berikut belum diset di .env: {', '.join(missing)}")
        
        # Pastikan folder data ada
        if not os.path.exists(cls.DATA_DIR):
            os.makedirs(cls.DATA_DIR)
            print(f"Created data directory at: {cls.DATA_DIR}")

# Instansiasi settings agar bisa di-import langsung
settings = Settings()

# Validasi saat module di-load
try:
    settings.validate()
    print("✅ Configuration loaded successfully.")
except ValueError as e:
    print(f"❌ Configuration Error: {e}")