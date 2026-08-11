import uvicorn
import logging
import sys
from fastapi import FastAPI

app = FastAPI()

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        logging.error("Failed to start the server: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()