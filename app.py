import streamlit as st
from microservices.llamaWork import LLMGenerator
from microservices.orchestrator import Orchestrator

# default settings
CROSS_ENCODER_NAME = "BAAI/bge-reranker-v2-m3"
EMBEDDER_NAME = "jinaai/jina-embeddings-v2-base-en"
DEFAULT_HF_MODEL = "Qwen/Qwen3-8B" 

@st.cache_resource
def load_orchestrator(config):
    """
    Initializes the Retriever and LLM. 
    """
    print(f"Loading Orchestrator with {config['type']} model...")
    orch = Orchestrator()
    orch.initVectorStore(EMBEDDER_NAME, CROSS_ENCODER_NAME)
    
    if config["type"] == "HuggingFace":
        orch.initHFLLM(config["model_id"])
    else:
        orch.initAPILLM(
            api_key=config['api_key'], 
            model_name=config['model_name'], 
            base_url=config['base_url']
        )
    return orch

if "model_config" not in st.session_state:
    st.title("PubMed RAG Setup")
    
    tab1, tab2 = st.tabs(["Hugging Face Model", "API Model"])
    
    with tab1:
        hf_model_id = st.text_input("Hugging Face Model ID", value=DEFAULT_HF_MODEL, placeholder="e.g., Qwen/Qwen3-8B")
        if st.button("Start with Hugging Face Model"):
            st.session_state.model_config = {"type": "HuggingFace", "model_id": hf_model_id}
            st.rerun()
            
    with tab2:
        st.subheader("API Connection Details")
        api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
        base_url = st.text_input("Base URL", placeholder="https://api.openai.com/v1 or similar")
        model_name = st.text_input("Model Name", placeholder="e.g., gpt-4, llama-3...")
        
        if st.button("Start with API Model"):
            if not api_key or not base_url or not model_name:
                st.error("Please fill in all API details.")
            else:
                st.session_state.model_config = {
                    "type": "API",
                    "api_key": api_key,
                    "base_url": base_url,
                    "model_name": model_name
                }
                st.rerun()
    st.stop()

theOrchestrator = load_orchestrator(st.session_state.model_config)
print("loaded the orchestrator!")

with st.sidebar:
    st.divider()
    st.header("Admin Actions")
    if st.button("Download Papers from S3"):
        with st.status("Connecting to S3...", expanded=True) as status:
            st.write("Initializing S3 Worker...")
            theOrchestrator.downloadFilesFromS3()
            st.write("Downloading new PDF files...")
            status.update(label="Download Complete!", state="complete", expanded=False)
        st.success("Dataset updated. Please restart/refresh if indexing is needed.")
        
    if st.button("Generate VectorStore"):
        with st.status("Building Vector Store...", expanded=True) as status:
            st.write("Loading documents...")
            theOrchestrator.recreateVectorStore()
            status.update(label="Vector Store Created Successfully!", state="complete", expanded=False)
        st.success("The new knowledge base is ready.")
    
    if st.button("Show Current PMC IDs"):
        try:
            with open(theOrchestrator.pathFile, "r") as f:
                ids = f.read()
                st.text_area("PMC IDs", ids, height=200)
        except FileNotFoundError:
            st.error(f"File {theOrchestrator.pathFile} not found.")

    st.header("Add Single Paper")
    pmc_id_input = st.text_input("Enter PMC ID (e.g., PMC9656789)")
    if st.button("Download & Index"):
        if not pmc_id_input:
            st.warning("Please enter a valid PMC ID.")
        else:
            with st.status("Processing Request...", expanded=True) as status:
                success = theOrchestrator.addEntryToVectorStore(pmc_id_input)
                if success:
                    status.update(label="Paper added successfully!", state="complete", expanded=False)
                    st.success(f"Paper {pmc_id_input} added to knowledge base.")
                else:
                    status.update(label="Failed to add paper", state="error", expanded=False)
                    st.error(f"Could not find or download PMC ID: {pmc_id_input}. Please check the ID and try again.")

st.title("PubMed RAG")

use_rag = st.checkbox("Use RAG", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = theOrchestrator.generateResponse(prompt, history=st.session_state.messages, use_rag=use_rag)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
