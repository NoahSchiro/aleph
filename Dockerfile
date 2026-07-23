FROM public.ecr.aws/lambda/python:3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY backend.py pyproject.toml uv.lock ${LAMBDA_TASK_ROOT}
RUN uv sync --frozen --only-group serve --no-dev

CMD ["backend.handler"]
