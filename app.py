import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page config
st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="💻")
st.title("💻 Corporate IT Helpdesk Chatbot")
st.caption("Ask about Shared Drives, Outlook issues, Laptop Warranty, or IT Peripherals.")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    #st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    st.markdown("**Sample Questions:**")
    st.markdown("- *How do I map the shared network drive on Windows?*")
    st.markdown("- *Outlook is stuck on 'Disconnected'. What should I do?*")
    st.markdown("- *How do I check my Dell laptop warranty?*")
    st.markdown("- *My mouse is broken. How do I request a replacement?*")

if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# Cache the vector store so it doesn't rebuild on every chat message
@st.cache_resource
def initialize_rag_chain():
    file_path = "it_helpdesk_knowledge_base.txt"
    if not os.path.exists(file_path):
        st.error(f"Could not find `{file_path}`. Make sure it is in the same directory as `app.py`.")
        st.stop()

    # 1. Load the text document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n===", "\n---", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Create Embeddings and FAISS Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. Initialize LLM and Prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    system_prompt = (
        "You are a helpful internal Corporate IT Helpdesk Assistant. "
        "Answer the employee's question strictly using the provided IT Knowledge Base context. "
        "Provide clear step-by-step instructions where applicable. "
        "If the answer is not in the context, tell the user to raise a ticket on the IT Service Portal "
        "or contact the IT Helpdesk directly—do not make up policies.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# Initialize chain
rag_chain = initialize_rag_chain()

# Manage chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your IT Helpdesk Assistant. How can I help you today?"}
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_query := st.chat_input("Type your IT issue here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching IT Knowledge Base..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)

            # Optional: Expandable box to verify which chunks were retrieved
            with st.expander("View Retrieved Policy Context"):
                for i, doc in enumerate(response["context"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})