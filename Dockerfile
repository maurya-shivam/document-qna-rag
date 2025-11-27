# Stage 1: Builder - Install dependencies
FROM python:3.9-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev

WORKDIR /document-qna

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runner - Create the final lightweight image
FROM python:3.9-alpine

WORKDIR /document-qna

# Copy only installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

COPY . /document-qna/

EXPOSE 5007

CMD python -m streamlit run run.py --server.address 0.0.0.0 --server.port 5007
