import shutil
from fastapi import FastAPI, Request, Depends, HTTPException, Cookie
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from app.gemini import GeminiLLM
from app.retriever import create_file_vectorstore, get_relevant_chunks
from fastapi.responses import JSONResponse, RedirectResponse
from google.oauth2 import id_token
from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleRequest
from typing import Optional
import time
import uuid
from app.supabase_client import supabase
from app.retriever import get_relevant_chunks, create_file_vectorstore
from fastapi import File, UploadFile, Form
import PyPDF2
import docx
import tempfile
import os
from typing import List
from fastapi import Path as FastAPIPath
from app.config import PERSONALITY_PROMPTS
from pathlib import Path

app = FastAPI()
llm = GeminiLLM()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  
        "http://localhost:3000", 
        "http://127.0.0.1:5173",  
    ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    question: str
    session_id: Optional[str] = None
    personality: Optional[str] = None  
    system_prompt: Optional[str] = None
    has_file: Optional[bool] = False

VITE_DEV_SERVER_URL = os.getenv("VITE_DEV_SERVER_URL")
VITE_BACKEND_URL = os.getenv("VITE_BACKEND_URL")
print(f"VITE_DEV_SERVER_URL: {VITE_DEV_SERVER_URL}")
@app.get("/")
async def root():
    return {"message": "Welcome to the Gemini LLM API!"}

@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    personality: str = Form(default="sage"),
    token: Optional[str] = Cookie(None)
):
    try:
        print(f"📁 Received file: {file.filename}, size: {file.size}")
        print(f"📍 Session ID: {session_id}")
        print(f"🎭 Personality: {personality}") 

        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
            clock_skew_in_seconds=60
        )
        user_id = idinfo.get("sub")
        print(f"👤 User ID: {user_id}")

        # Validate session_id
        if not session_id or session_id == 'null':
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Extract text from file
        file_content = await extract_text_from_file(file)
        print(f"📄 Extracted text length: {len(file_content)}")
        
        if not file_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")

        app_data = Path.home() / "AppData" / "Local" / "RAG_Chatbot" / "chroma_db_uploads"
        app_data.mkdir(parents=True, exist_ok=True)
        
        persist_directory = app_data / f"user_{user_id}_session_{session_id}"
        
        # Remove existing directory safely
        if persist_directory.exists():
            shutil.rmtree(persist_directory)
            print(f"🗑️ Cleared existing directory: {persist_directory}")

        # Ensure directory exists
        if persist_directory.exists():
            try:
                shutil.rmtree(persist_directory)
                print(f"🗑️ Cleared existing directory: {persist_directory}")
            except PermissionError:
                # If can't delete, create with timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                persist_directory = app_data / f"user_{user_id}_session_{session_id}_{timestamp}"

        persist_directory.mkdir(parents=True, exist_ok=True)
        # Split text into chunks
        chunks = split_text_into_chunks(file_content)
        print(f"📊 Created {len(chunks)} chunks")
        
        # Create vector store
        vectorstore = create_file_vectorstore(chunks, user_id, session_id)
        print(f"✅ Vector store created successfully")

        return {
            "message": "File uploaded successfully",
            "chunks_created": len(chunks),
            "filename": file.filename,
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/auth/google/url")
async def google_auth_url():
    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    if not CLIENT_ID:   
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not set in environment variables.")
    REDIRECT_URI = f'{VITE_BACKEND_URL}/auth/callback'
    SCOPES = ["openid", "email", "profile"]

    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"response_type=code&"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri={REDIRECT_URI}&"
        f"scope={'+'.join(SCOPES)}"
    )
    return {"url": auth_url}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/auth/callback")
def auth_callback(request: Request, code: str, state: Optional[str] = None):
    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    if not CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not set in environment variables.")
    CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    if not CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_SECRET not set in environment variables.")
    REDIRECT_URI = f"{VITE_BACKEND_URL}/auth/callback"

    try:
        # Step 1: Exchange code for tokens
        token_res = requests.post("https://oauth2.googleapis.com/token", 
            data={
                "code": code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=10
        )

        if token_res.status_code != 200:
            print(f"Token exchange failed: {token_res.status_code} - {token_res.text}")
            return RedirectResponse(url=f"{VITE_DEV_SERVER_URL}/login?error=token_exchange_failed")

        token_json = token_res.json()
        id_token_str = token_json.get("id_token")
        
        if not id_token_str:
            print("No id_token in response:", token_json)
            return RedirectResponse(url=f"{VITE_DEV_SERVER_URL}/login?error=no_id_token")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                idinfo = id_token.verify_oauth2_token(
                    id_token_str,
                    GoogleRequest(),
                    CLIENT_ID,
                    clock_skew_in_seconds=60
                )
                print(f"Token verification successful on attempt {attempt + 1}")
                break
                
            except ValueError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Token verification failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Token verification failed after {max_retries} attempts: {e}")
                    return RedirectResponse(url=f"{VITE_DEV_SERVER_URL}/login?error=token_verification_failed")

        # Success - set cookie and redirect
        response = RedirectResponse(url=f"{VITE_DEV_SERVER_URL}/google")
        response.set_cookie(
            key="token", 
            value=id_token_str, 
            httponly=True, 
            secure=False,
            max_age=3600,
            samesite="lax"
        )
        return response

    except requests.RequestException as e:
        print(f"Network error during OAuth: {e}")
        return RedirectResponse(url=f"{VITE_DEV_SERVER_URL}/login?error=network_error")
    except Exception as e:
        print(f"Unexpected error during OAuth: {e}")
        return RedirectResponse(url=f'{VITE_DEV_SERVER_URL}/login?error=unexpected_error')

from fastapi.responses import RedirectResponse

@app.get("/protected")
async def protected_route(token: Optional[str] = Cookie(None)):
    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID") 
    
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            CLIENT_ID,
            clock_skew_in_seconds=60
        )
        return {"message": f"Hello {idinfo['name']}!", "user": idinfo}
    except ValueError as e:
        print(f"Token validation failed: {str(e)}")
        return RedirectResponse(url="/login", status_code=302)


@app.post("/logout")
async def logout():
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie(
        key="token",
        httponly=True,
        secure=False,
        samesite="lax"
    )
    return response

@app.delete("/chat/delete/{session_id}")
async def delete_chat_session(
    session_id: str, 
    personality: str = Query(...),
    token: Optional[str] = Cookie(None)
):
    try:
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
            clock_skew_in_seconds=60
        )
        user_id = idinfo.get("sub")

        result = supabase.table("chat_messages") \
            .delete() \
            .eq("session_id", session_id) \
            .eq("user_id", user_id) \
            .eq("personality", personality) \
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return {"message": "Chat session deleted successfully", "deleted_count": len(result.data)}

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/chat/history")
async def get_chat_history(
    session_id: str, 
    personality: str = Query(...),
    token: Optional[str] = Cookie(None)
):
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        if not token:

            raise HTTPException(status_code=401, detail="Authentication required")

        
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
            clock_skew_in_seconds=60
        )
        user_id = idinfo.get("sub")
        
        result = supabase.table("chat_messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .eq("user_id", user_id) \
            .eq("personality", personality) \
            .order("created_at") \
            .execute()

        
        return {"messages": result.data}

    except ValueError as e:
        
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions")
async def get_user_sessions(
    personality: str = Query(...),
    token: Optional[str] = Cookie(None)
):
    try:
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
            clock_skew_in_seconds=60
        )
        user_id = idinfo.get("sub")

        result = supabase.table("chat_messages") \
            .select("session_id, message, created_at") \
            .eq("user_id", user_id) \
            .eq("is_user", True) \
            .eq("personality", personality) \
            .order("created_at", desc=True) \
            .execute()

        sessions = {}
        for msg in result.data:
            session_id = msg["session_id"]
            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "title": msg["message"][:50] + "..." if len(msg["message"]) > 50 else msg["message"],
                    "timestamp": msg["created_at"],
                    "personality": personality
                }

        return {"sessions": list(sessions.values())}

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def extract_text_from_file(file: UploadFile) -> str:
    content = await file.read()
    
    if file.filename and file.filename.endswith('.pdf'):
        return extract_pdf_text(content)
    elif file.filename and file.filename.endswith('.docx'):
        return extract_docx_text(content)
    elif file.filename and file.filename.endswith('.txt'):
        return content.decode('utf-8')
    else:
        raise ValueError("Unsupported file type or filename is missing")

def extract_pdf_text(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file_path = tmp_file.name  
    try:
        text = ""
        with open(tmp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    finally:
        try:
            os.unlink(tmp_file_path)
        except PermissionError:
            import time
            time.sleep(0.1)
            try:
                os.unlink(tmp_file_path)
            except:
                print(f"Warning: Could not delete temporary file {tmp_file_path}")

def extract_docx_text(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file_path = tmp_file.name  

    try:
        doc = docx.Document(tmp_file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    finally:
        try:
            os.unlink(tmp_file_path)
        except PermissionError:
            import time
            time.sleep(0.1)
            try:
                os.unlink(tmp_file_path)
            except:
                print(f"Warning: Could not delete temporary file {tmp_file_path}")


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

@app.delete("/delete-file/{session_id}")
async def delete_uploaded_file(session_id: str, token: Optional[str] = Cookie(None)):
    try:
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
            clock_skew_in_seconds=60
        )
        user_id = idinfo.get("sub")

        # ✅ Delete the vector store directory
        persist_directory = f"./chroma_db_uploads/user_{user_id}_session_{session_id}"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="No file found for this session")

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(query: QueryModel, token: Optional[str] = Cookie(None)):
    try:
        user_id = None
        if token:
            try:
                idinfo = id_token.verify_oauth2_token(
                    token,
                    GoogleRequest(),
                    os.getenv("GOOGLE_CLIENT_ID"),
                    clock_skew_in_seconds=60
                )
                user_id = idinfo.get("sub")
            except ValueError:
                user_id = "anonymous"
        else:
            user_id = "anonymous"
         
        session_id = query.session_id or str(uuid.uuid4())
        personality = query.personality or "sage"
        system_prompt = PERSONALITY_PROMPTS.get(personality, PERSONALITY_PROMPTS["sage"])

        if query.has_file:
            chunks = get_relevant_chunks(query.question, user_id, session_id)
        else:
            chunks = []
        
        if query.has_file:
            chunks = get_relevant_chunks(query.question, user_id, session_id)
        else:
            chunks = []
        
        # Generate response
        if chunks:
            context = "\n\n".join(chunks)
            prompt = f"""You are {personality.title()}, but you must prioritize the uploaded document content.  
IMPORTANT: Answer ONLY based on the context below. If the question cannot be answered from the context, say "I cannot find that information in the uploaded document."
Context from uploaded document:
{context}

User Question: {query.question}

Answer as {personality.title()} using ONLY the information from the context above."""
        else:
            prompt = f"""{system_prompt}

User Question: {query.question}
Note: No document context available. Please respond according to your personality."""

        answer = llm.generate(prompt)
        
        # Save messages
        user_message = {
            "user_id": user_id,
            "session_id": session_id,
            "message": query.question,
            "is_user": True,
            "personality": personality,
        }
        supabase.table("chat_messages").insert(user_message).execute()

        ai_message = {
            "session_id": session_id,
            "user_id": user_id,
            "message": answer,
            "is_user": False,
            "personality": personality,
        }
        supabase.table("chat_messages").insert(ai_message).execute()

        return {"answer": answer, "session_id": session_id}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": f"Error: {str(e)}"})
