from supabase import create_client, Client
import os
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")  
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if SUPABASE_URL is None or SUPABASE_KEY is None:
    raise ValueError("Missing Supabase environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
