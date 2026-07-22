"""
Main FastAPI Application

This is the entry point for the FastAPI application.
It configures CORS, includes routers, and sets up the app.

Key Features:
- CORS configuration for frontend access
- Automatic API documentation (Swagger UI and ReDoc)
- Organized route structure
- Production-ready error handling
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

# Create FastAPI application instance
app = FastAPI(
    title="Todo API",
    description="A simple Todo API built with FastAPI for Azure deployment",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # ReDoc at /redoc
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the frontend to communicate with the backend
# even though they're on different domains/ports

# Allowed origins - add your frontend URLs here
origins = [
    "http://localhost:5173",  # Local Vite dev server
    "http://localhost:3000",  # Alternative local port
    "http://localhost:8080",  # Another common dev port
    # Add your Azure Static Web App URL here after deployment:
    # "https://your-static-web-app.azurestaticapps.net",
]

# Add CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies and authentication
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include the router from routes.py
# This adds all the routes defined in routes.py to the main app
app.include_router(router)


# Optional: Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Code to run when the application starts.
    Useful for:
    - Initializing database connections
    - Loading configuration
    - Setting up background tasks
    """
    print("🚀 Todo API is starting up...")
    print("📝 API Documentation available at:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Code to run when the application shuts down.
    Useful for:
    - Closing database connections
    - Cleaning up resources
    - Saving state
    """
    print("👋 Todo API is shutting down...")


# This allows running the app directly with `python -m app.main`
# For development only - use uvicorn for production
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (development only)
    )
