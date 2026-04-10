import sys
sys.path.insert(0, "/content")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from email_triage_env.server.environment import EmailTriageEnvironment

app = FastAPI()
env = EmailTriageEnvironment()

class ActionInput(BaseModel):
    urgency: str = "not_urgent"
    category: str = "general"
    tone: str = "formal"

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs, "reward": None, "done": False}

@app.post("/step")
def step(action: ActionInput):
    obs = env.step(action.dict())
    return {
        "observation": obs,
        "reward": obs.get("last_reward"),
        "done": obs.get("done", False)
    }

@app.get("/state")
def state():
    return env.state

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy", "description": "Classify 3 emails", "max_steps": 3},
            {"name": "medium", "description": "Classify 5 emails", "max_steps": 5},
            {"name": "hard", "description": "Classify 10 emails", "max_steps": 10},
        ],
        "action_schema": {
            "urgency": "urgent | not_urgent",
            "category": "billing | technical | general | complaint",
            "tone": "formal | empathetic | urgent"
        }
    }

@app.get("/grader")
def grader():
    return {"score": env.grader_score(), "task": env.task_name}
