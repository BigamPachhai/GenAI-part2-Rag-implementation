#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()

pdf_path = "document loaders/deeplearning.pdf"
if not os.path.exists(pdf_path):
    raise SystemExit(
        f"PDF not found at '{pdf_path}'. Update the path or add the file before running."
    )

data = PyPDFLoader(pdf_path)
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db_hf"
)

vectorstore.persist()
