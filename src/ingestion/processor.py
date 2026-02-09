import logging
from typing import List, Dict
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Bertanggung jawab untuk membersihkan (Cleaning) dan memecah (Chunking) dokumen.
    Mengubah Raw HTML -> Clean Text -> Document Chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Ukuran maksimal karakter per potongan.
            chunk_overlap: Jumlah karakter yang tumpang tindih antar potongan (agar konteks tidak putus).
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""], # Prioritas pemotongan: Paragraf > Baris > Kata
            length_function=len,
        )

    def clean_html(self, html_content: str) -> str:
        """
        Menghapus tag HTML, script, dan style.
        Mengembalikan teks bersih yang bisa dibaca manusia.
        """
        if not html_content:
            return ""

        try:
            # Menggunakan lxml untuk parsing lebih cepat (pastikan lxml terinstall via requirements.txt)
            soup = BeautifulSoup(html_content, "lxml")

            # Hapus elemen yang tidak relevan untuk requirements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.extract()

            # Ekstrak teks dengan separator baris baru untuk menjaga struktur (seperti tabel/list)
            text = soup.get_text(separator="\n")

            # Normalisasi whitespace (menghapus spasi berlebih)
            lines = (line.strip() for line in text.splitlines())
            # Gabungkan kembali baris yang tidak kosong
            clean_text = '\n'.join(chunk for chunk in lines if chunk)
            
            return clean_text

        except Exception as e:
            logger.warning(f"⚠️ Gagal membersihkan HTML sebagian: {e}")
            return html_content # Fallback ke raw jika error

    def process_documents(self, raw_docs: List[Dict]) -> List[Document]:
        """
        Orchestrator utama: Raw Dict -> Clean Text -> List of Documents.
        
        Args:
            raw_docs: List dictionary output dari ConfluenceLoader
        
        Returns:
            List[Document]: Objek Document standar LangChain siap untuk VectorDB.
        """
        processed_docs = []

        for doc in raw_docs:
            try:
                # 1. Cleaning
                clean_content = self.clean_html(doc.get("content_raw", ""))
                
                # Metadata yang akan ditempel di setiap chunk (penting untuk sitasi nanti)
                metadata = {
                    "source": doc.get("url"),
                    "title": doc.get("title"),
                    "page_id": doc.get("id"),
                    "space": doc.get("space_key"),
                    "type": "confluence_requirement"
                }

                # 2. Chunking
                # create_documents otomatis memecah teks dan menempelkan metadata ke setiap pecahan
                chunks = self.text_splitter.create_documents(
                    texts=[clean_content], 
                    metadatas=[metadata]
                )

                processed_docs.extend(chunks)
                logger.debug(f"📄 Page '{doc.get('title')}' diproses menjadi {len(chunks)} chunks.")

            except Exception as e:
                logger.error(f"❌ Error memproses dokumen ID {doc.get('id')}: {e}")
                continue
        
        logger.info(f"✅ Total chunk yang dihasilkan: {len(processed_docs)}")
        return processed_docs