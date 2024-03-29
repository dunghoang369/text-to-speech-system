# Ref: https://github.com/cloudevents/sdk-python#receiving-cloudevents
from fastapi import FastAPI, Request

app = FastAPI()


# Create an endpoint at http://localhost:8000
@app.post("/")
async def on_event(request: Request):
    # Inspect cloud events
    response = await request.json()
    print(
        f"Received a new event: {response}"
    )

    # Return no content
    return "", 204
