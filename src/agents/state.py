import operator
from typing import Annotated, List, TypedDict, Union, Dict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    State merepresentasikan 'isi kepala' agen saat ini.
    LangGraph akan mengoper objek ini antar node.
    """
    
    # Messages: Riwayat percakapan (User + AI + Tool Outputs).
    # operator.add: Agar pesan baru ditambahkan ke list, bukan menimpa pesan lama.
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Context: Potongan dokumen requirement yang ditemukan dari Confluence (Vector DB).
    context: List[str]
    
    # References: Metadata dokumen yang digunakan sebagai referensi (source, title, URL, dll)
    references: List[Dict[str, str]]
    
    # Status: Untuk debugging/conditional logic (opsional)
    current_step: str