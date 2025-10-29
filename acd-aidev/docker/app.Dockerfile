FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app_streamlit /app/app_streamlit

ENV API_URL=http://localhost:8000
EXPOSE 8501
CMD ["streamlit", "run", "app_streamlit/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
