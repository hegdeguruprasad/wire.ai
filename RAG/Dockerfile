# Use the official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy only selected files
COPY server.py epsilonRAG.py requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
