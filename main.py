import sys
import logging
from typing import List
from langchain_core.messages import HumanMessage, BaseMessage
from src.agents.graph import app
from src.config import settings

# --- Setup Logging agar output terminal bersih ---
# Kita hanya ingin melihat log INFO dari agent kita, bukan debug log library lain
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("src.agents")
logger.setLevel(logging.INFO)

def print_banner():
    print("""
    =======================================================
       🚀 BRI AGENTIC AI - TRICENTIS TOOLS POC
    =======================================================
       Modul    : QTest & Confluence Integration
       Status   : Ready
       Exit     : Ketik 'exit' atau 'quit'
    =======================================================
    """)

def process_user_input(user_input: str, chat_history: List[BaseMessage]):
    """
    Mengirim input user ke LangGraph dan menangani streaming output.
    """
    print("\n🤖 Agent sedang berpikir...", end="", flush=True)

    # Siapkan state awal dengan history percakapan
    # Kita tambahkan pesan baru user ke history yang sudah ada
    current_messages = chat_history + [HumanMessage(content=user_input)]
    
    inputs = {"messages": current_messages}

    # app.stream() memungkinkan kita melihat proses langkah demi langkah
    # daripada menunggu lama sampai selesai semua.
    final_response = ""
    document_references = []  # Simpan referensi dokumen yang ditemukan
    last_agent_message = None  # Simpan pesan terakhir dari agent
    
    try:
        for event in app.stream(inputs):
            for node_name, value in event.items():
                
                # 1. Jika node 'retrieve' selesai
                if node_name == "retrieve":
                    print("\n📘 [Context] Dokumen relevan ditemukan dari Knowledge Base.")
                    
                    # Simpan referensi dokumen yang digunakan
                    if "references" in value and value["references"]:
                        document_references = value["references"]
                        print("\n📚 [Referensi Dokumen yang Ditemukan]:")
                        for idx, ref in enumerate(document_references, 1):
                            print(f"   {idx}. {ref.get('title', 'Unknown Title')}")
                            if ref.get('source'):
                                print(f"      🔗 URL: {ref['source']}")
                            if ref.get('space'):
                                print(f"      📁 Space: {ref['space']}")
                        print()  # Baris kosong untuk spacing
                
                # 2. Jika node 'agent' selesai (LLM memberikan respon teks / putusan tool)
                elif node_name == "agent":
                    last_msg = value["messages"][-1]
                    last_agent_message = last_msg  # Simpan untuk history
                    
                    # Cek apakah agent ingin memanggil tool
                    if last_msg.tool_calls:
                        tool_name = last_msg.tool_calls[0]['name']
                        print(f"\n🛠️  [Action] Agent memutuskan untuk menggunakan tool: '{tool_name}'...")
                    else:
                        # Jika tidak ada tool, berarti ini jawaban akhir
                        final_response = last_msg.content

                # 3. Jika node 'tools' selesai (Hasil eksekusi qTest)
                elif node_name == "tools":
                    # Ambil output dari tool message terakhir
                    tool_output = value["messages"][-1].content
                    print(f"✅ [Result] Hasil Eksekusi Tool:\n   {tool_output}")

    except Exception as e:
        print(f"\n❌ Terjadi Error: {e}")
        return chat_history

    # Tampilkan jawaban akhir Agent ke user
    if final_response:
        print("\n" + "-"*50)
        print(f"🤖 AGENT:\n{final_response}")
        print("-" * 50)
        
        # Tampilkan referensi dokumen yang digunakan untuk menghasilkan output
        if document_references:
            print("\n📚 [Referensi yang Digunakan untuk Menghasilkan Test Case]:")
            for idx, ref in enumerate(document_references, 1):
                print(f"   {idx}. {ref.get('title', 'Unknown Title')}")
                if ref.get('source'):
                    print(f"      🔗 URL: {ref['source']}")
                if ref.get('space'):
                    print(f"      📁 Space: {ref['space']}")
            print("-" * 50)
        
        # Update history dengan jawaban agent agar percakapan nyambung
        # (Di implementation production, history ini disimpan di database/Redis)
        if last_agent_message:
            current_messages.append(last_agent_message)
    
    return current_messages

def main():
    # Validasi Config sebelum mulai
    try:
        print(f"🔗 Menghubungkan ke: {settings.CONFLUENCE_URL} & {settings.QTEST_URL}...")
    except ValueError as e:
        print(f"❌ Config Error: {e}")
        sys.exit(1)

    print_banner()

    # Inisialisasi memori percakapan sesi ini
    chat_history = []

    while True:
        try:
            user_input = input("\nUser (Anda) > ")
            
            if not user_input.strip():
                continue
                
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Sampai jumpa! Menutup sesi.")
                break

            # Proses input
            chat_history = process_user_input(user_input, chat_history)

        except KeyboardInterrupt:
            print("\n👋 Force Close.")
            break

if __name__ == "__main__":
    main()