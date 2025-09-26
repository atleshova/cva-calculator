# Dockerfile for deploying Streamlit CVA SCVA prototype on Google Cloud Run or other container platforms

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY Streamlit_CVA_SCVA_App.py ./

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "Streamlit_CVA_SCVA_App.py", "--server.port=8501", "--server.address=0.0.0.0"]