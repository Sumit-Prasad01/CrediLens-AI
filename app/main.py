import uvicorn
from app.api.app import app

if __name__ == "__main__":
    uvicorn.run("app.api.app:app", host="127.0.0.1", port=8000, reload=True)
