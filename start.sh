#!/bin/bash

# Start the backend API (in the background)
echo "Starting backend API..."
cd "$(dirname "$0")"
python app_with_cors.py &
BACKEND_PID=$!

# Wait a bit for the API to start
sleep 2

# Start the frontend
echo "Starting frontend..."
cd stock-analyzer-frontend
npm run dev -- --port 8001 --host 0.0.0.0

# When the frontend is closed, also stop the backend
kill $BACKEND_PID