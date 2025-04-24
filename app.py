# =============================================================================
# quillstack: Streamlit + LangChain RAG Chatbot for Multiple PDFs
# =============================================================================
import os                              # operating-system interactions for paths and environment
from dotenv import load_dotenv        # function to load .env file variables into os.environ
import streamlit as st                # Streamlit library for building the web interface
from PyPDF2 import PdfReader          # PdfReader to read and extract text from PDF files

# LangChain components:
from langchain.text_splitter import RecursiveCharacterTextSplitter  # utility to split text into chunks
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings  # spaCy embeddings for text
from langchain_community.vectorstores import FAISS             # FAISS vector store for similarity search
from langchain_openai import ChatOpenAI                        # OpenAI LLM wrapper
from langchain_core.prompts import ChatPromptTemplate          # prompt template for the conversation
from langchain.tools.retriever import create_retriever_tool    # tool to wrap a retriever as a LangChain tool
from langchain.agents import AgentExecutor, create_tool_calling_agent  # agent and executor for tool calling

# -----------------------------------------------------------------------------
# 1) Load environment variables (API keys)
# -----------------------------------------------------------------------------
load_dotenv()                         # load variables from .env into os.environ
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # retrieve OpenAI key from environment
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # retrieve Anthropic key if using Claude

# -----------------------------------------------------------------------------
# 2) Initialize embeddings model (spaCy)
# -----------------------------------------------------------------------------
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")  # create a spaCy embeddings instance

# -----------------------------------------------------------------------------
# 3) Function: Read and concatenate PDF text
# -----------------------------------------------------------------------------
def pdf_read(pdf_files):                                             # define function to read PDFs
    """
    Given a list of uploaded PDF file-like objects, extract text from each
    page and return the combined string of all pages.
    """
    combined = ""                                               # initialize empty string for all text
    for pdf in pdf_files:                                         # loop through each uploaded file
        reader = PdfReader(pdf)                                  # create a PdfReader instance
        for page in reader.pages:                                # iterate through each page
            page_text = page.extract_text() or ""             # extract text, handle None
            combined += page_text                               # append page text to combined
    return combined                                              # return the full concatenated text

# -----------------------------------------------------------------------------
# 4) Function: Split text into chunks
# -----------------------------------------------------------------------------
def get_chunks(text, chunk_size=1000, chunk_overlap=0):               # define chunking function
    """
    Split the input text into chunks up to `chunk_size` characters,
    with optional overlap between chunks.
    """
    splitter = RecursiveCharacterTextSplitter(                     # initialize text splitter
        chunk_size=chunk_size,                                     # set max characters per chunk
        chunk_overlap=chunk_overlap                                # set overlap between chunks
    )
    return splitter.split_text(text)                              # return list of text chunks

# -----------------------------------------------------------------------------
# 5) Function: Build and save FAISS index
# -----------------------------------------------------------------------------
def build_vector_store(chunks, persist_dir="faiss_db"):           # define function to build FAISS index
    """
    Embed each text chunk into vectors using spaCy embeddings and
    save the FAISS index to the local `persist_dir` directory.
    """
    db = FAISS.from_texts(chunks, embedding=embeddings)           # create FAISS index from chunks
    db.save_local(persist_dir)                                    # save index locally

# -----------------------------------------------------------------------------
# 6) Function: Create and run the conversational agent
# -----------------------------------------------------------------------------
def get_conversational_chain(retriever_tool, question):              # define function for QA chain
    """
    Create a LangChain agent that wraps a retriever tool and an LLM,
    invoke it on the user's question, and return the answer.
    """
    # 6.1 Initialize the LLM (GPT-4.0 Turbo)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",                                # specify the LLM model
        temperature=0,                                              # deterministic responses
        openai_api_key=OPENAI_API_KEY                               # pass the API key
    )

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([                     # build the conversation prompt
        ("system",
         "You are a helpful assistant. Use context; if you cannot answer, "
         "reply 'answer is not available in the context'."),      # system instructions
        ("placeholder", "{chat_history}"),                    # placeholder for past chats
        ("human", "{input}"),                                  # placeholder for user input
        ("placeholder", "{agent_scratchpad}")                  # placeholder for agent scratchpad
    ])

    # Create tool-calling agent
    agent = create_tool_calling_agent(
        llm=llm,                                                   # the LLM instance
        tools=[retriever_tool],                                    # list containing the retriever tool
        prompt=prompt                                              # the prompt template
    )
    executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)  # wrap in executor

    # Invoke agent on the question
    result = executor.invoke({"input": question})               # run the agent
    return result["output"]                                     # return the agent's output

# -----------------------------------------------------------------------------
# 7) Streamlit UI
# -----------------------------------------------------------------------------
def main():                                                        # define the main Streamlit function
    st.set_page_config(page_title="QuillStack PDF Chatbot")      # set page title
    st.header("📚 Chat with Your PDFs")                          # display header

    # Sidebar: upload & index
    with st.sidebar:                                               # begin sidebar context
        st.subheader("1) Upload PDF(s)")                          # sidebar section title
        uploaded = st.file_uploader(                               # file uploader widget
            "Select one or more PDFs", type="pdf", accept_multiple_files=True
        )
        if st.button("2) Process & Index"):                      # button to trigger processing
            if not uploaded:                                      # if no files uploaded
                st.warning("🔺 Please upload at least one PDF.") # show warning
            else:
                with st.spinner("Indexing PDF content..."):   # show spinner while processing
                    text = pdf_read(uploaded)                     # read PDF text
                    chunks = get_chunks(text)                    # split into chunks
                    build_vector_store(chunks)                   # build FAISS index
                st.success("✅ Indexed successfully!")           # show success message

    # Main: ask questions
    query = st.text_input("Ask a question about your PDFs")       # text input for questions
    if query:                                                     # if user has entered a query
        faiss_db = FAISS.load_local(
            "faiss_db", 
            embeddings,
            allow_dangerous_deserialization=True                  #opt-in to unpickle the data
            )      # load FAISS index
        retriever = faiss_db.as_retriever()                       # create retriever
        tool = create_retriever_tool(                             # wrap retriever as tool
            retriever,
            name="pdf_retriever",
            description="Fetches relevant PDF passages"
        )
        answer = get_conversational_chain(tool, query)            # get answer from agent
        st.write("**Answer:**", answer)                        # display the answer

# -----------------------------------------------------------------------------
# 8) Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":                                       # only run when executed directly
    main()                                                         # call the main function

