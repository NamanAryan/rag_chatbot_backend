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

def find_user_session_directory(base_dir: Path, user_id: str, session_id: str) -> Optional[Path]:
    """Find directory with or without timestamp"""
    # First try exact match (backward compatibility)
    exact_path = base_dir / f"user_{user_id}_session_{session_id}"
    if exact_path.exists():
        return exact_path
    
    # Search for timestamped versions
    pattern = f"user_{user_id}_session_{session_id}_*"
    matching_dirs = list(base_dir.glob(pattern))
    
    if matching_dirs:
        # Return the most recent one
        latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
        print(f"🔍 Found timestamped directory: {latest_dir}")
        return latest_dir
    
    print(f"❌ No directory found for user_{user_id}_session_{session_id}")
    return None

def get_relevant_chunks(query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[str]:
    if not user_id or not session_id or session_id == 'null':
        print(f"Invalid parameters: user_id={user_id}, session_id={session_id}")
        return []
    
    try:
        # Set base directory based on environment
        if os.getenv("RENDER"):
            base_dir = Path("/tmp/chroma_db_uploads")
        else:
            base_dir = Path("./chroma_db_uploads")
        
        # ✅ FIXED: Use directory finder to handle timestamped directories
        persist_directory = find_user_session_directory(base_dir, user_id, session_id)
        
        print(f"🔍 Looking for directory: {persist_directory}")
        
        if not persist_directory or not persist_directory.exists():
            print(f"Directory doesn't exist: {persist_directory}")
            # Debug: Show available directories
            print(f"📁 Available directories in {base_dir}:")
            for dir_path in base_dir.glob(f"user_{user_id}_session_*"):
                print(f"  - {dir_path.name}")
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
        import traceback
        traceback.print_exc()
        return []

def create_file_vectorstore(chunks: List[str], user_id: str, session_id: str, persist_directory: Optional[str] = None):
    """Create vector store with explicit persist directory"""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Use provided persist_directory or fallback to default logic
    if not persist_directory:
        if os.getenv("RENDER"):
            persist_directory = f"/tmp/chroma_db_uploads/user_{user_id}_session_{session_id}"
        else:
            persist_directory = f"./chroma_db_uploads/user_{user_id}_session_{session_id}"
    
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore
