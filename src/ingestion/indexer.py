import os
import shutil
import logging
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorIndexer:
    """
    Bertanggung jawab mengelola ChromaDB (Vector Database).
    Fungsi: Embedding -> Storing -> Retrieving.
    """

    def __init__(self):
        # Menggunakan OpenAI Embeddings untuk mengubah teks menjadi vektor
        # Ini memungkinkan pencarian semantik (makna), bukan hanya keyword.
        self.embedding_model = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small" # Model efisien & murah dari OpenAI
        )
        self.persist_directory = settings.CHROMA_DB_PATH
        self.collection_name = settings.CHROMA_COLLECTION_NAME

    def create_index(self, documents: List[Document], reset_db: bool = False):
        """
        Membuat atau memperbarui index vector database.
        
        Args:
            documents: List dokumen yang sudah di-chunk dari processor.py
            reset_db: Jika True, hapus DB lama dan buat baru (mencegah duplikat saat testing).
        """
        if not documents:
            logger.warning("⚠️ Tidak ada dokumen untuk di-index.")
            return

        if reset_db:
            self._clear_database()

        logger.info(f"💾 Sedang menyimpan {len(documents)} chunks ke Vector DB...")
        
        try:
            # Chroma.from_documents otomatis melakukan:
            # 1. Call OpenAI Embeddings API untuk setiap chunk
            # 2. Simpan vektor + metadata ke folder lokal
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logger.info(f"✅ Sukses! Data tersimpan di: {self.persist_directory}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Gagal membuat index: {e}")
            raise

    def get_retriever(self, k: int = 4):
        """
        Mengembalikan objek 'Retriever' yang bisa dipakai oleh Agent LangGraph.
        Agent akan menggunakan ini untuk mencari info relevan.
        
        Args:
            k: Jumlah dokumen teratas yang diambil saat pencarian.
        """
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
        )
        
        # search_type="mmr" (Maximal Marginal Relevance) bagus untuk
        # mendapatkan hasil yang relevan tapi beragam (tidak redundan).
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k}
        )

    def _clear_database(self):
        """Hapus folder database fisik untuk reset bersih."""
        if os.path.exists(self.persist_directory):
            logger.warning(f"🧹 Menghapus database lama di {self.persist_directory}...")
            shutil.rmtree(self.persist_directory)