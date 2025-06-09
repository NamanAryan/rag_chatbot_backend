# RAG Chatbot with Google Sign-In

This project implements a RAG (Retrieval-Augmented Generation) chatbot with Google Sign-In authentication.

## Google Sign-In Implementation Guide

### Prerequisites

1. Python 3.8 or higher
2. A Google Cloud Project
3. OAuth 2.0 credentials from Google Cloud Console

### Setup Steps

1. **Create Google Cloud Project and OAuth Credentials**

   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google+ API
   - Go to "Credentials" and create OAuth 2.0 Client ID
   - Add authorized redirect URIs (e.g., `http://localhost:8000/auth/callback`)
   - Download the client credentials JSON file

2. **Install Required Dependencies**

   ```bash
   pip install google-auth google-auth-oauthlib google-auth-httplib2
   ```

3. **Environment Variables**
   Create a `.env` file in your project root:

   ```
   GOOGLE_CLIENT_ID=your_client_id
   GOOGLE_CLIENT_SECRET=your_client_secret
   GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback
   ```

4. **Implementation Steps**

   a. Create a new file `app/auth.py`:

   ```python
   from google.oauth2.credentials import Credentials
   from google_auth_oauthlib.flow import Flow
   from google.auth.transport.requests import Request
   import os
   import json

   # Initialize OAuth2 flow
   def create_flow():
       client_config = {
           "web": {
               "client_id": os.getenv("GOOGLE_CLIENT_ID"),
               "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
               "auth_uri": "https://accounts.google.com/o/oauth2/auth",
               "token_uri": "https://oauth2.googleapis.com/token",
               "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI")]
           }
       }
       return Flow.from_client_config(
           client_config,
           scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
           redirect_uri=os.getenv("GOOGLE_REDIRECT_URI")
       )
   ```

   b. Update `app/main.py` to include authentication routes:

   ```python
   from fastapi import FastAPI, Request
   from fastapi.responses import RedirectResponse
   from .auth import create_flow

   app = FastAPI()

   @app.get("/login")
   async def login():
       flow = create_flow()
       authorization_url, state = flow.authorization_url()
       return RedirectResponse(authorization_url)

   @app.get("/auth/callback")
   async def auth_callback(request: Request):
       flow = create_flow()
       flow.fetch_token(
           authorization_response=str(request.url)
       )
       credentials = flow.credentials
       # Store credentials in session or database
       return RedirectResponse("/")
   ```

5. **Frontend Integration**
   Add a login button to your frontend:
   ```html
   <a href="/login" class="google-login-button"> Sign in with Google </a>
   ```

### Security Considerations

1. Always use HTTPS in production
2. Store credentials securely
3. Implement proper session management
4. Add CSRF protection
5. Set appropriate session timeouts

### Testing

1. Run your application locally
2. Click the "Sign in with Google" button
3. Complete the Google authentication flow
4. Verify that you're redirected back to your application

### Troubleshooting

- Ensure all environment variables are set correctly
- Check that redirect URIs match exactly
- Verify API is enabled in Google Cloud Console
- Check application logs for detailed error messages

## Additional Resources

- [Google OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [Google Cloud Console](https://console.cloud.google.com/)
