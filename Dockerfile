# Create a custom docker image to upload to AWS so that Lambda can pull it down
FROM public.ecr.aws/lambda/python:3.12

COPY backend.py ${LAMBDA_TASK_ROOT}
RUN pip install fastapi mangum numpy polars boto3

CMD ["backend.handler"]
