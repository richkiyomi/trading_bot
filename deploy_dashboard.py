"""
Modal deployment script for Streamlit P&L Dashboard.

Deploy with: modal deploy deploy_dashboard.py
Access URL will be shown after deployment.
"""
import modal

# Create Modal app
app = modal.App("iron-condor-dashboard")

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "pandas>=2.0.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.24.0",
        "starlette>=0.27.0",
    )
    .add_local_file("models.py", "/root/models.py")
    .add_local_file("main.py", "/root/main.py")
)

# Secrets - only need database credentials for dashboard
secrets = [
    modal.Secret.from_name("postgres-secret"),
]

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
)
@modal.asgi_app()
def serve():
    """Serve Streamlit app via ASGI."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    from starlette.applications import Starlette
    from starlette.responses import StreamingResponse
    from starlette.routing import Route
    import httpx
    import threading
    import time
    
    # Start Streamlit server in background
    streamlit_script = Path("/root/main.py")
    
    def run_streamlit():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(streamlit_script),
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true",
                "--server.enableCORS=false",
                "--server.enableXsrfProtection=false",
                "--browser.gatherUsageStats=false",
                "--server.runOnSave=false",
            ],
            check=False,  # Don't fail if process exits
        )
    
    # Start Streamlit in background thread
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Wait for Streamlit to start
    time.sleep(8)  # Give Streamlit more time to fully initialize
    
    # Proxy requests to Streamlit
    async def proxy(request):
        """Proxy requests to Streamlit server."""
        # Build full URL including path and query string
        path = request.url.path
        query = str(request.url.query)
        url = f"http://localhost:8501{path}"
        if query:
            url += f"?{query}"
        
        try:
            # Get request body
            body = await request.body()
            
            # Prepare headers (exclude problematic ones)
            headers = {}
            for k, v in request.headers.items():
                k_lower = k.lower()
                if k_lower not in ['host', 'content-length', 'connection', 'upgrade']:
                    headers[k] = v
            
            # Add necessary headers for Streamlit
            headers['Host'] = 'localhost:8501'
            
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.request(
                    request.method,
                    url,
                    content=body,
                    headers=headers,
                )
                
                # Get response headers
                response_headers = dict(response.headers)
                # Remove headers that shouldn't be forwarded
                response_headers.pop('content-encoding', None)
                response_headers.pop('transfer-encoding', None)
                response_headers.pop('connection', None)
                
                # Return response with proper content
                return StreamingResponse(
                    iter([response.content]),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("content-type"),
                )
        except httpx.ConnectError:
            from starlette.responses import Response
            return Response(
                "Streamlit is starting up. Please wait a few seconds and refresh.",
                status_code=503
            )
        except Exception as e:
            from starlette.responses import Response
            return Response(
                f"Error connecting to Streamlit: {str(e)}",
                status_code=503
            )
    
    app = Starlette(routes=[Route("/{path:path}", proxy)])
    return app
