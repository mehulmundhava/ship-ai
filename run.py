"""
Application Runner

Simple script to run the FastAPI application.
Usage: python run.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=3009, reload=True)

