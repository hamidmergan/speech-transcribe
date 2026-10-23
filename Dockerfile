FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir ".[ui]" \
    && useradd --create-home --uid 1000 app

USER app
EXPOSE 8000

# Drop a recording in the browser and get the transcript back as text, VTT, SRT or JSON.
# Settings come from SPEECH_TRANSCRIBE_* environment variables,
# for example SPEECH_TRANSCRIBE_BASE_URL and SPEECH_TRANSCRIBE_MODEL.
CMD ["speech-transcribe", "serve", "--host", "0.0.0.0", "--port", "8000"]
