from fastapi import Path as FastAPIPath
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional
import os
from pathlib import Path

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder="/app/cache/sentence_transformers"
)

def get_page_contents(results):
    return [result.page_content for result in results]

def get_relevant_chunks(query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[str]:
    if not user_id or not session_id or session_id == 'null':
        print(f"Invalid parameters: user_id={user_id}, session_id={session_id}")
        return []
    
    try:
        app_data = Path.home() / "AppData" / "Local" / "RAG_Chatbot" / "chroma_db_uploads"
        persist_directory = app_data / f"user_{user_id}_session_{session_id}"
        
        print(f"🔍 Looking for directory: {persist_directory}")
        
        if not persist_directory.exists():
            print(f"Directory doesn't exist: {persist_directory}")
            return []
        
        db = Chroma(
            persist_directory=str(persist_directory), 
            embedding_function=embeddings
        )
        
        results = db.similarity_search_with_relevance_scores(query, k=5)
        print(f"Found {len(results)} results for query: {query}")
        
        documents = [doc for doc, score in results if score > 0.1]
        contents = get_page_contents(documents)
        print(f"Returning {len(contents)} relevant chunks")
        return contents
        
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []



def create_file_vectorstore(documents: List[str], user_id: str, session_id: str, personality: str = "sage"):
    try:
        app_data = Path.home() / "AppData" / "Local" / "RAG_Chatbot" / "chroma_db_uploads"
        persist_directory = app_data / f"user_{user_id}_session_{session_id}"
        persist_directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating vector store at: {persist_directory}")

        vectorstore = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings
        )

        for i, chunk in enumerate(documents):
            vectorstore.add_texts(
                texts=[chunk],
                metadatas=[{
                    "chunk_index": i,
                    "user_id": user_id,
                    "session_id": session_id,
                    "personality": personality
                }],
                ids=[f"chunk_{i}"]
            )
        
        print(f"Vector store created with {len(documents)} chunks")
        return vectorstore
        
    except Exception as e:
        print(f"Error in create_file_vectorstore: {e}")
        raise Exception(f"Vector store creation failed: {str(e)}")


