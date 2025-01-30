sudo apt install redis-server
celery -A tasks worker --loglevel=INFO
