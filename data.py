from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load PDF documents
loader = PyPDFLoader("income tax-1961.pdf")
docs = loader.load()

# Step 2: Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

# Step 3: Create embeddings and save them to a vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(final_documents, embeddings)

# Step 4: Save the vector store locally
vector_store.save_local("income_tax_vectorstore")
print("Vector store saved!")
