FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml README.md ./
COPY app ./app
COPY src ./src
COPY configs ./configs
COPY validation ./validation
COPY data ./data
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501

CMD ["bash", "scripts/run_demo_stack.sh"]
