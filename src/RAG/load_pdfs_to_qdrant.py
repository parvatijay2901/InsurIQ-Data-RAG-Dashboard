import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from insuriq_utils import INSURIQ_CACHE  # local cache path

# Configurations
pdf_folder_path = "data/raw/PDFs/"  # Path to folder containing PDF files
qdrant_collection = "insurance_documents"  # Qdrant collection name
qdrant_path = INSURIQ_CACHE / qdrant_collection  # Local path to store Qdrant index

# Setup Embedding Model
# Using a lightweight sentence-transformer model for generating document embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

# load PDF documents
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyMuPDFLoader(pdf_path)
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents from PDF files.")

# Initialize Qdrant Client
client = QdrantClient(path=str(qdrant_path))

# If cached Qdrant collection already exists, remove it for a clean rebuild
if qdrant_path.exists():
    print("Removing cached Qdrant data...")
    shutil.rmtree(qdrant_path)


# Store Documents in Qdrant
print(
    f"Creating new Qdrant collection '{qdrant_collection}' from {len(documents)} documents"
)

# Embed and store the documents locally in Qdrant
qdrant = Qdrant.from_documents(
    documents=documents,
    embedding=embeddings_model,
    path=str(qdrant_path),
    collection_name=qdrant_collection,
)

qdrant.client.close()