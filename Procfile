release: python train_lite.py
web: gunicorn -k uvicorn.workers.UvicornWorker api_app:app --bind 0.0.0.0:$PORT