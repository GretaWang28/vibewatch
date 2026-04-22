FROM python:3.11-slim

# Install nginx
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY main.py .
COPY preprocess.py .
COPY recommender.py .
COPY netflix_data.csv .
COPY static/ ./static/
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY start.sh .
RUN chmod +x start.sh

# Remove default nginx site
RUN rm -f /etc/nginx/sites-enabled/default

EXPOSE 80

CMD ["./start.sh"]
