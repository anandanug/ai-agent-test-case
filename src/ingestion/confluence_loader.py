import logging
from typing import List, Dict, Optional
from atlassian import Confluence
from src.config import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfluenceLoader:
    """
    Bertanggung jawab untuk menghubungkan dan mengambil data raw dari Confluence.
    Updated: Sekarang mendukung pengambilan metadata Attachments.
    """

    def __init__(self):
        try:
            self.confluence = Confluence(
                url=settings.CONFLUENCE_URL,
                username=settings.CONFLUENCE_USERNAME,
                password=settings.CONFLUENCE_API_TOKEN,
                cloud=True
            )
            logger.info(f"✅ Terhubung ke Confluence: {settings.CONFLUENCE_URL}")
        except Exception as e:
            logger.error(f"❌ Gagal terhubung ke Confluence: {e}")
            raise

    def load_from_space(self, space_key: str, limit: int = 50) -> List[Dict]:
        logger.info(f"📥 Memulai pengambilan dokumen dari Space: {space_key}...")
        try:
            pages = self.confluence.get_all_pages_from_space(
                space=space_key,
                start=0,
                limit=limit,
                expand='body.storage,version',
                content_type='page'
            )
            
            documents = []
            for page in pages:
                # Ambil attachments untuk setiap halaman
                attachments = self._get_page_attachments(page.get("id"))
                doc = self._parse_page_structure(page, attachments)
                documents.append(doc)
            
            logger.info(f"✅ Berhasil mengambil {len(documents)} dokumen dari Space {space_key}.")
            return documents
        except Exception as e:
            logger.error(f"❌ Error saat mengambil data space {space_key}: {e}")
            return []

    def load_specific_page(self, page_id: str) -> Optional[Dict]:
        """
        Mengambil satu halaman spesifik + attachments nya.
        """
        try:
            page = self.confluence.get_page_by_id(
                page_id=page_id, 
                expand='body.storage,version'
            )
            
            # [BARU] Ambil info lampiran
            attachments = self._get_page_attachments(page_id)
            
            logger.info(f"✅ Berhasil mengambil page ID: {page_id} dengan {len(attachments)} lampiran.")
            return self._parse_page_structure(page, attachments)
        except Exception as e:
            logger.error(f"❌ Gagal mengambil page ID {page_id}: {e}")
            return None

    def _get_page_attachments(self, page_id: str) -> List[Dict]:
        """
        [BARU] Mengambil daftar file yang dilampirkan pada halaman.
        """
        try:
            attachments = self.confluence.get_attachments_from_content(
                page_id=page_id,
                start=0,
                limit=50
            )
            results = attachments.get('results', [])
            return results
        except Exception:
            logger.warning(f"⚠️ Gagal mengambil attachment untuk page {page_id}")
            return []

    def _parse_page_structure(self, page_data: Dict, attachments: List[Dict] = []) -> Dict:
        """
        Menggabungkan teks halaman dengan daftar lampiran agar terbaca oleh AI.
        """
        base_url = settings.CONFLUENCE_URL.rstrip('/')
        webui = page_data.get('_links', {}).get('webui', '')
        page_url = f"{base_url}{webui}"
        
        raw_content = page_data.get("body", {}).get("storage", {}).get("value", "")

        # [BARU] Format Attachments menjadi teks agar masuk ke Vector DB
        # Ini memungkinkan AI berkata: "Silakan cek file X di link Y"
        if attachments:
            attachment_text = "\n\n<h3>📂 Attachments (Lampiran):</h3><ul>"
            for att in attachments:
                title = att.get("title", "Unknown File")
                download_link = f"{base_url}{att.get('_links', {}).get('download', '')}"
                attachment_text += f"<li>File: {title} - <a href='{download_link}'>Download Link</a></li>"
            attachment_text += "</ul>"
            
            # Tempelkan info attachment ke bawah konten asli
            raw_content += attachment_text

        return {
            "id": page_data.get("id"),
            "title": page_data.get("title"),
            "space_key": page_data.get("space", {}).get("key", "UNKNOWN"),
            "content_raw": raw_content,
            "url": page_url,
            "version": page_data.get("version", {}).get("number", 1),
            "source_type": "confluence"
        }