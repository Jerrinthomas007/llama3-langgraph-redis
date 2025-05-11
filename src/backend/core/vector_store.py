from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.backend.core.config import QDRANT_HOST, QDRANT_PORT

embedding_model = HuggingFaceEmbeddings()

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
qdrant = Qdrant(client=qdrant_client, collection_name="chatbot-memory", embeddings=embedding_model)