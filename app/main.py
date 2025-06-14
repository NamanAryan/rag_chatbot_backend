from fastapi import FastAPI, Request, Depends, HTTPException, Cookie
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from app.gemini import GeminiLLM
from app.retriever import get_relevant_chunks
from fastapi.responses import JSONResponse, RedirectResponse
from google.oauth2 import id_token
from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleRequest
from typing import Optional
import time
import uuid
from app.supabase_client import supabase

app = FastAPI()
llm = GeminiLLM()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Gemini LLM API!"}

@app.get("/auth/google/url")
async def google_auth_url():
    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    if not CLIENT_ID:   
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not set in environment variables.")
    REDIRECT_URI = "http://localhost:8000/auth/callback"
    SCOPES = ["openid", "email", "profile"]

    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"response_type=code&"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri={REDIRECT_URI}&"
        f"scope={'+'.join(SCOPES)}"
    )
    return {"url": auth_url}

@app.get("/auth/callback")
def auth_callback(request: Request, code: str, state: Optional[str] = None):
    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    if not CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not set in environment variables.")
    CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    if not CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_SECRET not set in environment variables.")
    REDIRECT_URI = "http://localhost:8000/auth/callback"

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
            return RedirectResponse(url="http://localhost:5173/login?error=token_exchange_failed")

        token_json = token_res.json()
        id_token_str = token_json.get("id_token")
        
        if not id_token_str:
            print("No id_token in response:", token_json)
            return RedirectResponse(url="http://localhost:5173/login?error=no_id_token")

        # Step 2: Verify token with retry logic
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
                    return RedirectResponse(url="http://localhost:5173/login?error=token_verification_failed")

        # Success - set cookie and redirect
        response = RedirectResponse(url="http://localhost:5173/google")
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
        return RedirectResponse(url="http://localhost:5173/login?error=network_error")
    except Exception as e:
        print(f"Unexpected error during OAuth: {e}")
        return RedirectResponse(url="http://localhost:5173/login?error=unexpected_error")

@app.get("/protected")
async def protected_route(token: Optional[str] = Cookie(None)):
    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID") 
    
    if not token:
        raise HTTPException(status_code=401, detail="No token found")
    
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            CLIENT_ID,
            clock_skew_in_seconds=60
        )
        return {"message": f"Hello {idinfo['name']}!", "user": idinfo}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

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
async def delete_chat_session(session_id: str, token: Optional[str] = Cookie(None)):
    try:
        # Verify user is authenticated
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get user ID from token
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
            clock_skew_in_seconds=60
        )
        user_id = idinfo.get("sub")

        # Delete all messages for this session and user
        result = supabase.table("chat_messages") \
            .delete() \
            .eq("session_id", session_id) \
            .eq("user_id", user_id) \
            .execute()

        # Check if any rows were deleted
        if not result.data:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return {"message": "Chat session deleted successfully", "deleted_count": len(result.data)}

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_chat_history(session_id: str, token: Optional[str] = Cookie(None)):
    try:
        print(f"Received request for session_id: {session_id}")  # ✅ Add debug
        
        # Verify user is authenticated
        if not token:
            print("No token provided")  # ✅ Add debug
            raise HTTPException(status_code=401, detail="Authentication required")

        print(f"Token: {token[:20]}...")  # ✅ Add debug (partial token)
        
        idinfo = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            os.getenv("GOOGLE_CLIENT_ID"),
            clock_skew_in_seconds=60
        )
        user_id = idinfo.get("sub")
        print(f"User ID: {user_id}")  # ✅ Add debug

        print("Querying Supabase...")  # ✅ Add debug
        result = supabase.table("chat_messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .eq("user_id", user_id) \
            .order("created_at") \
            .execute()

        print(f"Supabase result: {len(result.data)} messages")  # ✅ Add debug
        return {"messages": result.data}

    except ValueError as e:
        print(f"ValueError: {e}")  # ✅ Add debug
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(f"Exception: {e}")  # ✅ Add debug
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions")
async def get_user_sessions(token: Optional[str] = Cookie(None)):
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
            .order("created_at", desc=True) \
            .execute()

        sessions = {}
        for msg in result.data:
            session_id = msg["session_id"]
            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "title": msg["message"][:50] + "..." if len(msg["message"]) > 50 else msg["message"],
                    "timestamp": msg["created_at"]
                }
        return {"sessions": list(sessions.values())}

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(query: Query, token: Optional[str] = Cookie(None)):
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
        
        chunks = get_relevant_chunks(query.question)
        if not chunks:
            return {"answer": "No relevant information found."}
        context = "\n\n".join(chunks)
        prompt = f"""Use the context below to answer the question.
    
Context:
{context}

Question:
{query.question}
"""
        answer = llm.generate(prompt)

        user_message = {
            "user_id": user_id,
            "session_id": session_id,
            "message": query.question,
            "is_user": True,
        }
        supabase.table("chat_messages").insert(user_message).execute()

        ai_message = {
            "session_id": session_id,
            "user_id": user_id,
            "message": answer,
            "is_user": False,
        }
        supabase.table("chat_messages").insert(ai_message).execute()

        return {"answer": answer, "session_id": session_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": f"Error processing request: {str(e)}"})
