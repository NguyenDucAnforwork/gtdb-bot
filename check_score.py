from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from config import settings

# 1. Setup
embeddings = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding") # Hoặc model bạn đang dùng
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# 2. Embed câu hỏi
query = "Chở trẻ em 8 tuổi không đội mũ bảo hiểm khi đi xe máy bị phạt bao nhiêu?"
query_vector = embeddings.embed_query(query)

# 3. Search thô (không threshold)
results = client.search(
    collection_name=settings.QDRANT_COLLECTION_NAME,
    query_vector=query_vector,
    limit=3,
    with_payload=True
)

# 4. In kết quả
for hit in results:
    print(f"Score: {hit.score} | Content: {hit.payload.get('text', '')[:100]}...")