from openai import AsyncOpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os 

class RAGPipeline:
    load_dotenv()

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv('AICREDITS_API_KEY'),
            base_url='https://api.aicredits.in/v1'
        )

        self.vector_db_client = QdrantClient(':memory:')

        self.vector_db_client.create_collection(
            collection_name='gift_of_the_magi',
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )

    def load_documents(self) -> str: 
        filepath = '/Users/princesingh/Downloads/rag_text.txt'

        try: 
            # Reading the file here, reason I didn't use .readlines() because
            # it returns a list and I need it to return a string
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            # This is just a check, in Python empty string is False
            # if its empty, it will just raise an error that file is empty.
            if not content:
                raise ValueError(f'File is empty or blank.')
            
            return content
        
        except FileExistsError:
            raise FileNotFoundError(f'File does not exist.')
        
    def chunk_document(self, text: str) -> list[str]:
        start_marker = '*** START OF THE PROJECT GUTENBERG EBOOK THE GIFT OF THE MAGI ***'
        end_marker = '*** END OF THE PROJECT GUTENBERG EBOOK THE GIFT OF THE MAGI ***'

        # start_pos finds the starting point of start_marker + 
        # length of start_marker = the complete start_marker

        # end_post: for end_post we don't need such complications
        # it gives us the starting point where the end_marker starts and we just need that
        start_pos = text.find(start_marker) + len(start_marker)
        end_pos = text.find(end_marker)

        filtered_text = text[start_pos:end_pos].strip() # Here we just filtered out only the story/content we needed by slicing. strip() just removes any trailing whitespaces.

        paragraph_split = filtered_text.split('\n\n') # Here we filtered or splitted the text we just filtered into paragraphs. In simple text files, paragraphs which means double enter is '\n\n'.

        paragraphs = [p.strip() for p in paragraph_split if p.strip()] # We are going through each splitted paragraph and removing any trailing whitespaces but before doing that we're checking if after applying .strip(), does it still contain any content, if not reject it and move forward, thus removing empty paragraphs.

        # Chunking logic starts from here 
        chunk_size = 500 
        overlap = 100
        chunks = []
        current_chunk = ''

        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size: # In this condition, we're checking if the current_chunk which initially will be empty + len of current paragraph is smaller than chunk_size which is 500, if yes, save it in the current_chunk
                current_chunk += para + '\n\n'
            else:
                if current_chunk.strip(): # In here we're just storing the current_chunk into the main chunk list only if after stripping of whitespaces it still contains some content.
                    chunks.append(current_chunk.strip())

                overlap_text = current_chunk[-overlap:] # This is an interesting part, so in chunking, we just can't chop the paragraphs, if we did that it will lose its meaning, para1 contains some other info, para2 something different context, so we use this to copy the last few 100 characters from para1 and paste it into the starting of para2

                space_index = overlap_text.find('\n')

                if space_index != -1:
                    overlap_text = overlap_text[space_index + 1:]

                if len(para) > chunk_size: # This check is to check if the paragraph itself isn't too big
                    for i in range(0, len(para), chunk_size - overlap):
                        chunks.append(para[i:i + chunk_size].strip())
                    current_chunk = ''

                else:
                    current_chunk = overlap_text + para + '\n\n'

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        chunks = [chunk for chunk in chunks if chunk.strip()]

        min_size = 150

        chunks = [chunk for chunk in chunks if len(chunk) >= min_size]

        # This completes the chunking logic here, now we are just making sure that each chunk is unique and not repeating
        seen = set()
        unique = []

        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique.append(chunk)

        chunks = unique

        return chunks
    
    async def embedding(self, texts: list[str]) -> list[dict]:
        response = await self.client.embeddings.create(
            model = 'text-embedding-3-small',
            input = texts
        )

        embedded_chunks = []

        for i, (chunk, embedding) in enumerate(zip(texts, response.data)):
            embedded_chunks.append({
                'id': i,
                'text': chunk,
                'embedding': embedding.embedding
            })

        return embedded_chunks
    
    def store_embeddings(self, embedded_chunks: list[dict]):
        points = []
        for chunk in embedded_chunks:
            points.append(
                PointStruct(
                    id = chunk['id'],
                    vector = chunk['embedding'],
                    payload = {'text': chunk['text']}
                )
            )
        
        self.vector_db_client.upsert(
            collection_name='gift_of_the_magi',
            points=points
        )

    async def main(self):
        document = self.load_documents()
        chunks = self.chunk_document(document)
        embedded_chunks = await self.embedding(chunks)
        self.store_embeddings(embedded_chunks)
        