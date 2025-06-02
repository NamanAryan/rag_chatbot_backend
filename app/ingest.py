import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
import os

DATA_PATH = "./documents"
CHROMA_PATH = "chroma_db"

def main ():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents() -> list:
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def save_to_chroma(documents: list):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  
    
    Chroma.from_documents(
        documents,
        HuggingFaceEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    #db.persist()
    print(f"Saved {len(documents)} documents to ChromaDB at {CHROMA_PATH}.")


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
    return chunks


if __name__ == "__main__":
    main()


