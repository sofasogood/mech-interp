FROM python:3.11-slim

RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx

WORKDIR /app
COPY fastapi_router.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "fastapi_router:app", "--host", "0.0.0.0", "--port", "8080"]
