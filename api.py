from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ObsidianPyLabRequest(BaseModel):
    vaultPath: str
    notePath: Optional[str] = None
    text: Optional[str] = None


app = FastAPI()

origins = ["app://obsidian.md"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

available_fns = {"similar": lambda x: None}


@app.get("/")
def read_root():
    return {"scripts": [f"http://127.0.0.1:5000/function/{fn}" for fn in available_fns]}


@app.post("/similar")
def similar(request: ObsidianPyLabRequest):
    print(request)
    items = [{"path": f"Commitment.md", "name": "Commitment", "info": {"score": 0.25}}]

    return {"contents": items}
