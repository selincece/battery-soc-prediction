FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src
COPY app_streamlit /app/app_streamlit

ENV PYTHONPATH=/app/src
ENV API_URL=http://localhost:8000
ENV DATA_DIR=/app/data
ENV ARTIFACTS_DIR=/app/artifacts

EXPOSE 8501
CMD ["streamlit", "run", "app_streamlit/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
