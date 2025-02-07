# syntax=docker/dockerfile:1

FROM docker.io/library/python:3.11.9-slim-bookworm

# Basic packages installation
RUN apt-get update && \
    apt-get install -y curl bash vim nano libssl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Repo Files
COPY skyresh/* /skyresh/
WORKDIR /skyresh/

# Python Packages
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Expose the port
EXPOSE 8080

# Launch Uvicorn
# uvicorn application:app --host 0.0.0.0 --port 8080 --log-level info
# CMD ["sleep", "infinity"]
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]