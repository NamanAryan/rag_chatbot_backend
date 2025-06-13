from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_page_contents(results):
    return [result.page_content for result in results]

def get_relevant_chunks(query: str):
    db = Chroma(persist_directory="chroma_db", embedding_function=HuggingFaceEmbeddings())
    results = db.similarity_search_with_relevance_scores(query, k=3)
    documents = [doc for doc, score in results]  
    contents = get_page_contents(documents)      
    return contents
