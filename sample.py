import uvicorn
import logging
import sys
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, Friend! Use /people to get the list of names."}

# Separate object for people names


@app.get("/morning/{name}")
async def say_good_morning(name: str):
    return {"message": f"Good morning, {name}!"}

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8002)
    except Exception as e:
        logging.error("Failed to start the server: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()