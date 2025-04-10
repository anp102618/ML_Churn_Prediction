FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install
COPY Fast_API/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Fast_API app folder
COPY Fast_API ./Fast_API

# Copy the constants.yaml from root to container root
COPY constants.yaml ./constants.yaml

# Expose port
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "Fast_API.app:app", "--host", "0.0.0.0", "--port", "8000"]
