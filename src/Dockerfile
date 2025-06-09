FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
# Expose the port the app runs on
#EXPOSE 5000
# Set environment variables
#ENV FLASK_APP=app.py
#ENV FLASK_SECRET_KEY=61320b01ae5e0798393ef25f2dc72faf43703e60ba089b0