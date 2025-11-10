from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

def retrieve (query, client, model):
    # query = "how to deploy a contract on juno"
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name="juno_docs",
        query_vector=q_vec,
        limit=3,
    )

    for hit in results:
        print(hit.score, hit.payload)

    context = "\n\n".join([r.payload["text"] for r in results])
    prompt = f"""Use the following docs to answer the question.
    
    {context}
    
    Question: {query}
    """

    from openai import OpenAI
    client = OpenAI()
    answer = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
    )

    return (answer.choices[0].message.content)
