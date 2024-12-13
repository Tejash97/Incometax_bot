import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.title("Intelligent Income Tax Bot")

# Groq API Key and LLM setup
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    temperature=0.4,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question,
    if the context not in the context just say Its not there in the Income TAX 1961
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the precomputed vector store
if "vectors" not in st.session_state:
    st.session_state.vectors = FAISS.load_local(
        "income_tax_vectorstore",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True)
    st.write("Vector Store Loaded!")

# Input for user question
prompt1 = st.text_input("Enter Your Question From Income Tax-1961")

# Handle user query
if prompt1:
    if "vectors" in st.session_state:
        # Create the document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed_time = time.process_time() - start
        st.write("Response time:", elapsed_time)
        
        # Display the response
        st.write(response['answer'])

        # Display document similarity results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Vector Store is not initialized!")
