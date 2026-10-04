"""Run: uvicorn api:app --reload   then open http://localhost:8000/docs"""
from fastapi import FastAPI
from pydantic import BaseModel

from pipeline import admin, ask, office_user

app = FastAPI(title="Dossier Assistant")

class Query(BaseModel):
    question: str
    office: str | None = None     # DEMO ONLY: in production the PHP backend derives this from the JWT, never the client

@app.post("/ask")
def ask_endpoint(q: Query):
    user = admin if q.office is None else office_user(q.office)
    return ask(q.question, user)
