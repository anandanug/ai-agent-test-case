import requests
import json
import logging
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.config import settings

# Setup Logging
logger = logging.getLogger(__name__)

# --- 1. API Client Wrapper (Logika Komunikasi) ---

class QTestClient:
    def __init__(self):
        self.base_url = settings.QTEST_URL.rstrip('/')
        self.token = settings.QTEST_API_TOKEN
        self.project_id = settings.QTEST_PROJECT_ID
        # [BARU] Ambil default parent ID dari config
        self.default_parent_id = settings.QTEST_PARENT_MODULE_ID
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def create_test_case(self, title: str, description: str, steps: List[Any], parent_id: Optional[int] = None) -> Dict:
        url = f"{self.base_url}/api/v3/projects/{self.project_id}/test-cases"
        
        # [UPDATE] Gunakan self.default_parent_id jika parent_id kosong
        target_parent_id = parent_id if parent_id else self.default_parent_id
        
        # Pastikan tidak None/Kosong. Jika masih kosong, error akan terjadi lagi.
        if not target_parent_id:
             return {"error": "Module ID (Parent ID) belum diset di .env!"}

        # Validasi: Pastikan minimal 3 step
        if len(steps) < 3:
            logger.warning(f"⚠️ Test case '{title}' hanya memiliki {len(steps)} step. Direkomendasikan minimal 3-5 step untuk test case yang detail.")
        
        payload = {
            "name": title,
            "description": description,
            "parent_id": int(target_parent_id), # Pastikan dikirim sebagai integer
            "test_steps": []
        }

        # Loop untuk memproses semua steps
        for idx, step in enumerate(steps):
            if hasattr(step, "action"): 
                action_val = step.action
                expected_val = step.expected_result
            elif isinstance(step, dict):
                action_val = step.get("action", "")
                expected_val = step.get("expected_result", "")
            else:
                action_val = str(step)
                expected_val = ""

            payload["test_steps"].append({
                "description": action_val,
                "expected": expected_val,
                "order": idx + 1
            })
        
        # Kirim request setelah semua steps diproses
        try:
            logger.info(f"\n\n📤 Mengirim request create test case: {title} dengan {len(steps)} step(s)...")
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"\n✅ Test Case berhasil dibuat! ID: {data.get('id')}\n")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Gagal membuat test case di qTest: {e}")
            if e.response is not None:
                # Print detail error dari qTest untuk debugging
                logger.error(f"Response Content: {e.response.text}")
            return {"error": str(e)}

# --- 2. Tool Definition untuk LangGraph (Logika Agent) ---

# Mendefinisikan struktur input agar AI tahu format data yang harus dikirim
class TestStepInput(BaseModel):
    action: str = Field(description="Langkah spesifik yang dilakukan user (Contoh: 'Klik tombol Login')")
    expected_result: str = Field(description="Hasil yang diharapkan dari langkah tersebut (Contoh: 'User berhasil masuk ke dashboard')")

class CreateTestCaseInput(BaseModel):
    title: str = Field(description="Judul Test Case yang singkat dan jelas")
    description: str = Field(description="Deskripsi tujuan test case atau preconditions")
    steps: List[TestStepInput] = Field(
        description=(
            "Daftar langkah pengujian. ⚠️ PENTING: HARUS TERDIRI DARI MINIMAL 3-5 LANGKAH TERPISAH. "
            "JANGAN menggabungkan multiple aksi dalam 1 step. "
            "Setiap step harus fokus pada SATU aksi spesifik dengan expected result yang jelas. "
            "Contoh struktur: Step 1 (Setup/Login) → Step 2 (Navigate) → Step 3 (Action) → Step 4 (Verification) → Step 5 (Final Check). "
            "Untuk fitur kompleks, gunakan 5-7 step untuk coverage lengkap."
        )
    )
    
@tool("create_qtest_test_case", args_schema=CreateTestCaseInput)
def create_qtest_test_case_tool(title: str, description: str, steps: List[Any]) -> str:
    """
    Gunakan tool ini untuk MENYIMPAN test case ke dalam qTest Management System.
    
    ⚠️ PENTING SEBELUM MEMANGGIL TOOL INI:
    - Pastikan test case sudah dirancang dengan MINIMAL 3-5 langkah terpisah
    - Setiap step harus fokus pada SATU aksi spesifik (jangan gabungkan multiple aksi)
    - Setiap step harus memiliki expected result yang jelas
    - JANGAN membuat test case dengan hanya 1 step!
    
    Panggil tool ini HANYA SETELAH kamu selesai merancang SEMUA langkah-langkah testing secara detail.
    Output tool ini adalah ID dari test case yang berhasil dibuat.
    """
    # Validasi jumlah step sebelum membuat test case
    if len(steps) < 3:
        warning_msg = (
            f"⚠️ PERINGATAN: Test case '{title}' hanya memiliki {len(steps)} step. "
            f"Direkomendasikan minimal 3-5 step untuk test case yang detail dan dapat di-maintain. "
            f"Pertimbangkan untuk memecah step menjadi langkah-langkah yang lebih granular."
        )
        logger.warning(warning_msg)
        # Tetap lanjutkan pembuatan, tapi dengan warning
    
    client = QTestClient()
    
    result = client.create_test_case(title, description, steps)
    
    if "error" in result:
        return f"GAGAL membuat test case. Error: {result['error']}"
    
    tc_id = result.get("id")
    web_url = result.get("links", [{}])[0].get("href", "URL not found") # Simplifikasi pengambilan link
    
    success_msg = f"SUKSES! Test Case '{title}' berhasil dibuat dengan ID {tc_id} ({len(steps)} step(s)). Link: {web_url}"
    if len(steps) < 3:
        success_msg += f"\n⚠️ Catatan: Test case ini hanya memiliki {len(steps)} step. Untuk test case yang lebih detail, pertimbangkan untuk menambahkan lebih banyak step."
    
    return success_msg