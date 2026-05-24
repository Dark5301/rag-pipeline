from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel 
from pydantic_ai.providers.openai import OpenAIProvider 
from openai import AsyncOpenAI
from dotenv import load_dotenv
from RAG import RAGPipeline as rag 
import asyncio
import os

load_dotenv()

embedder = AsyncOpenAI(
    api_key = os.getenv('AICREDITS_API_KEY'),
    base_url = 'https://api.aicredits.in/v1'
)

pipeline = rag()

model = OpenAIChatModel(
    'gpt-4o-mini',
    provider = OpenAIProvider(
        api_key = os.getenv('AICREDITS_API_KEY'),
        base_url = 'https://api.aicredits.in/v1'
    )
)

agent = Agent(
    model = model,
    system_prompt = """You are a helpful assistant that answers questions 
    about the story 'The Gift of the Magi' by O. Henry.
    You have access to a retrieval tool that searches the story.
    Always use the tool to find relevant context before answering.
    Only answer based on what the tool returns — do not make things up."""
)

@agent.tool_plain
def retrieve_chunks(ctx: str) -> str:
    embedding = asyncio.run(query_embedding(ctx))
    chunks = chunk_retrieval(embedding)
    return '\n\n'.join(chunks)

async def query_embedding(user_query: str) -> list[float]:
    response = await embedder.embeddings.create(
        model = 'text-embedding-3-small',
        input = [user_query]
    )
    return response.data[0].embedding

def chunk_retrieval(query_embedding: list[float]) -> list[str]:
    top_k = 3
    results = pipeline.vector_db_client.query_points(
        collection_name = 'gift_of_the_magi',
        query = query_embedding,
        limit = top_k
    )
    return [result.payload['text'] for result in results.points]

async def main(query: str):
    result = await agent.run(query)
    print(result.output)

if __name__ == '__main__':
    asyncio.run(pipeline.main()) 
    
    queries = [
        "What did Della sell to buy Jim a gift?",
        "What did Jim sell to buy Della a gift?",
        "How did Della feel about having only $1.87?",
        "Why are Jim and Della called the magi at the end?",
        "What is the moral of the story?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        asyncio.run(main(query))