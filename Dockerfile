FROM python:3.12.4-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
