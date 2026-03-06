from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "debug_port": os.environ.get("PORT")}

@app.get("/")
def root():
    return HTMLResponse("<h1>Spare Parts Dashboard</h1><p>Skeleton mode active. If you see this, the server is running correctly.</p>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting skeleton app on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
