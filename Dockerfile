FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run gunicorn with 4 workers
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "4"]

EXPOSE 5000