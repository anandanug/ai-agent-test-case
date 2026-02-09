# python ingest_data.py --page_id 3905224707

import argparse
import logging
import sys
from src.ingestion.confluence_loader import ConfluenceLoader
from src.ingestion.processor import DocumentProcessor
from src.ingestion.indexer import VectorIndexer

# Setup logging agar terlihat rapi di terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main(space_key: str, page_id: str, limit: int, reset: bool):
    """
    Fungsi utama pipeline ingestion:
    1. Collect Sources (Confluence)
    2. Filter & Normalize (Processing)
    3. Store to Vector DB (Indexing)
    """
    logger.info("🚀 Memulai Pipeline Ingestion Data...")

    # --- 1. Inisialisasi Modul ---
    try:
        loader = ConfluenceLoader()
        processor = DocumentProcessor()
        indexer = VectorIndexer()
    except Exception as e:
        logger.error(f"❌ Gagal menginisialisasi modul: {e}")
        sys.exit(1)

    raw_docs = []

    # --- 2. Tahap Collect Sources (Extract) ---
    if page_id:
        logger.info(f"🔍 Mode: Mengambil satu halaman spesifik (ID: {page_id})...")
        doc = loader.load_specific_page(page_id)
        if doc:
            raw_docs.append(doc)
    elif space_key:
        logger.info(f"🔍 Mode: Mengambil dokumen dari Space '{space_key}' (Limit: {limit})...")
        raw_docs = loader.load_from_space(space_key, limit=limit)
    else:
        logger.error("❌ Error: Harap tentukan --space atau --page_id.")
        sys.exit(1)

    if not raw_docs:
        logger.warning("⚠️ Tidak ada dokumen yang ditemukan. Proses dihentikan.")
        sys.exit(0)

    # --- 3. Tahap Filter & Normalize (Transform) ---
    logger.info("⚙️ Sedang memproses dan membersihkan dokumen...")
    clean_documents = processor.process_documents(raw_docs)

    if not clean_documents:
        logger.warning("⚠️ Dokumen kosong setelah dibersihkan. Cek konten Confluence Anda.")
        sys.exit(0)

    # --- 4. Tahap Update Knowledge Store (Load) ---
    logger.info(f"💾 Menyimpan {len(clean_documents)} chunks ke Vector Database...")
    indexer.create_index(clean_documents, reset_db=reset)

    logger.info("✅ Pipeline Selesai! Data siap digunakan oleh Agent AI.")

if __name__ == "__main__":
    # Setup Argument Parser untuk Command Line Interface (CLI)
    parser = argparse.ArgumentParser(description="Tool Ingestion Data BRI POC")
    
    parser.add_argument("--space", type=str, help="Kunci Space Confluence (misal: 'DS', 'PROJ')")
    parser.add_argument("--page_id", type=str, help="ID Halaman spesifik (jika hanya ingin update 1 page)")
    parser.add_argument("--limit", type=int, default=10, help="Batasan jumlah halaman yang diambil (Default: 10)")
    parser.add_argument("--reset", action="store_true", help="Hapus database lama dan buat dari awal (Fresh Start)")

    args = parser.parse_args()

    # Validasi input minimal
    if not args.space and not args.page_id:
        parser.print_help()
        sys.exit(1)

    main(args.space, args.page_id, args.limit, args.reset)