import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.agents.state import AgentState
from src.ingestion.indexer import VectorIndexer
from src.tools.qtest_mcp import create_qtest_test_case_tool

# Setup Logging
logger = logging.getLogger(__name__)

# --- 1. SETUP KOMPONEN ---

# Inisialisasi LLM 
llm = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    temperature=0 # Temperature 0 agar hasil konsisten dan akurat untuk testing
)

# Inisialisasi Retriever (Akses ke Vector DB)
indexer = VectorIndexer()
retriever = indexer.get_retriever(k=4) # Ambil 4 potongan dokumen paling relevan

# Daftar Tools yang bisa dipakai Agent
tools = [create_qtest_test_case_tool]

# Bind tools ke LLM (Memberi tahu LLM bahwa dia punya cabang)
llm_with_tools = llm.bind_tools(tools)

# --- 2. DEFINISI NODE (Langkah-langkah kerja) ---

def retrieve_node(state: AgentState):
    logger.info("\n🔍 Agent sedang mencari informasi di Knowledge Base...")
    
    last_message = state["messages"][-1]
    query = last_message.content
    
    # Lakukan pencarian
    docs = retriever.invoke(query)
    
    # Format hasil pencarian menjadi string
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Ekstrak referensi (metadata) dari dokumen yang ditemukan
    # Menghindari duplikat berdasarkan source URL
    references = []
    seen_sources = set()
    
    for doc in docs:
        metadata = doc.metadata
        source = metadata.get("source", "")
        
        # Hanya tambahkan jika belum ada (menghindari duplikat)
        if source and source not in seen_sources:
            references.append({
                "title": metadata.get("title", "Unknown Title"),
                "source": source,
                "page_id": metadata.get("page_id", ""),
                "space": metadata.get("space", "")
            })
            seen_sources.add(source)
    
    return {
        "context": [context_text],
        "references": references,
        "current_step": "retrieving"
    }

def agent_node(state: AgentState):
    """
    Langkah 2: LLM berpikir.
    """
    logger.info("\n🤖 Agent sedang memproses informasi dan merancang test case...\n")
    
    system_prompt = (
        "Anda adalah QA Engineer Senior di Bank BRI. Tugas Anda adalah membuat Test Case yang SANGAT DETAIL.\n\n"
        
        "CONTEXT DARI CONFLUENCE:\n{context}\n\n"
        
        "ATURAN PEMBUATAN TEST CASE (WAJIB DIPATUHI):\n"
        "1. PENTING: Test case HARUS terdiri dari MINIMAL 3-5 langkah terpisah. JANGAN PERNAH membuat test case hanya dalam 1 step!\n"
        "2. Pecah setiap test case menjadi langkah-langkah atomik yang detail:\n"
        "   - Setiap step harus fokus pada SATU aksi spesifik\n"
        "   - Setiap step harus memiliki expected result yang jelas\n"
        "   - Gunakan pola: Setup → Action → Verification → Cleanup (jika perlu)\n\n"
        
        "CONTOH TEST CASE YANG BENAR (Notifikasi Email):\n"
        "Step 1: Login ke aplikasi dengan kredensial yang valid\n"
        "   Expected: User berhasil masuk ke dashboard utama\n"
        "Step 2: Masuk ke menu Settings > Notifications\n"
        "   Expected: Halaman pengaturan notifikasi ditampilkan\n"
        "Step 3: Aktifkan toggle 'Email Notifications'\n"
        "   Expected: Toggle berubah menjadi aktif (ON)\n"
        "Step 4: Pilih jenis notifikasi 'Transaction Alert' dari dropdown\n"
        "   Expected: Dropdown menampilkan pilihan yang dipilih\n"
        "Step 5: Klik tombol 'Save Settings'\n"
        "   Expected: Pesan sukses 'Settings saved successfully' muncul\n"
        "Step 6: Verifikasi email konfirmasi diterima di inbox\n"
        "   Expected: Email dengan subject 'Notification Settings Updated' diterima dalam 30 detik\n\n"
        
        "CONTOH TEST CASE YANG SALAH (JANGAN LAKUKAN INI):\n"
        "Step 1: Aktifkan notifikasi email dan verifikasi email diterima\n"
        "   SALAH: Ini menggabungkan banyak aksi dalam 1 step!\n\n"
        
        "3. Gunakan Bahasa Indonesia yang formal dan jelas.\n"
        "4. Pastikan 'Expected Result' diisi untuk SETIAP langkah, bukan hanya langkah terakhir.\n"
        "5. Untuk fitur yang kompleks, buat lebih banyak step (5-7 step) untuk memastikan coverage lengkap.\n"
        "6. SETELAH merancang SEMUA langkah-langkah tersebut, WAJIB panggil tool 'create_qtest_test_case' untuk menyimpannya.\n"
        "7. Jika user meminta 'test case positif', pastikan semua langkah menguji alur sukses dari awal sampai akhir."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Format context agar masuk ke prompt
    context_str = "\n".join(state["context"]) if state["context"] else "Tidak ada dokumen relevan ditemukan."
    
    chain = prompt | llm_with_tools
    response = chain.invoke({
        "messages": state["messages"], 
        "context": context_str
    })
    
    return {"messages": [response], "current_step": "reasoning"}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Router Logic: Menentukan langkah selanjutnya.
    Jika LLM ingin memanggil tool -> Pergi ke 'tools'.
    Jika LLM hanya membalas chat biasa -> Selesai (__end__).
    """
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        logger.info(f"🛠️ Agent memutuskan untuk menggunakan tool: {last_message.tool_calls[0]['name']}")
        return "tools"
    
    logger.info("\n✅ Agent selesai berpikir. Mengembalikan jawaban ke user.")
    return "__end__"

# --- 3. MENYUSUN GRAPH (LangGraph) ---

workflow = StateGraph(AgentState)

# Tambahkan Node
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools)) # Node bawaan LangGraph untuk eksekusi tools

# Tentukan Entry Point (Mulai dari mencari info)
workflow.set_entry_point("retrieve")

# Tambahkan Edges (Garis penghubung antar node)
workflow.add_edge("retrieve", "agent") # Habis cari info -> Berpikir

# Conditional Edge (Cabang keputusan)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",  # Jika perlu tool, ke node tools
        "__end__": END     # Jika selesai, stop
    }
)

# Setelah tool selesai dijalankan, kembali ke agent untuk konfirmasi (loop)
workflow.add_edge("tools", "agent")

# Compile menjadi aplikasi yang bisa dijalankan
app = workflow.compile()