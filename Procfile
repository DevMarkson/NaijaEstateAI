release: python train.py --data lagos-rent.csv
web: gunicorn -k uvicorn.workers.UvicornWorker api_app:app --bind 0.0.0.0:$PORT