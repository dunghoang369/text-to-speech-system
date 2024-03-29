FROM python:3.9

# Create a folder /app is the current working directory
WORKDIR /app

# Copy necessary files to app
COPY ./TextToSpeech.py /app

COPY ./requirements.txt /app

COPY ./tts /app/tts

COPY ./onnx_models /app/onnx_models

# Port will be exposed
EXPOSE 9000

# Install necessary libraries
RUN pip install -r requirements.txt --no-cache-dir

CMD ["uvicorn", "TextToSpeech:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]
