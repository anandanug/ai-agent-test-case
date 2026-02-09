from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.mcp_tools.confluence import get_confluence_page
from src.mcp_tools.qtest import create_test_case_in_qtest

def build_agent():
    # 1. Definisikan LLM (Otak Agent)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 2. Daftarkan Tools (MCP) yang bisa dipakai
    tools = [get_confluence_page, create_test_case_in_qtest]

    # 3. Buat Prompt (Instruksi Utama)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Anda adalah QA Automation Agent untuk BRI. "
                   "Tugas Anda membaca requirement dari Confluence, memahaminya, "
                   "lalu membuat Test Case yang sesuai di qTest."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # 4. Rakit Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor