#!/bin/bash

uvicorn application:app --host 0.0.0.0 --port 8080 --log-level info &
python gradio_app.py