# Placeholder Dockerfile for markov-rl-api-cache
# Customize base image, dependencies and entrypoint as needed.

FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
# Install runtime requirements (adjust as necessary)
# RUN pip install -r requirements.txt

CMD ["python", "-c", "print('markov-rl-api-cache placeholder')"]

