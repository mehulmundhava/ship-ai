#!/bin/bash

# Startup script for Ship RAG AI

echo "🚀 Starting Ship RAG AI..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your configuration."
    exit 1
fi

# Start the FastAPI server
echo "Starting FastAPI server on port 3009..."
uvicorn main:app --host 0.0.0.0 --port 3009 --reload

