FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY lr_model.pkl .
COPY scaler.pkl .

EXPOSE 5000

CMD ["python", "app.py"]