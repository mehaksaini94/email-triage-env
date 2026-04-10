from typing import Optional
from pydantic import BaseModel

class EmailTriageAction(BaseModel):
    urgency: str = "not_urgent"
    category: str = "general"
    tone: str = "formal"

class EmailTriageObservation(BaseModel):
    email_subject: str = ""
    email_body: str = ""
    task_name: str = "easy"
    step_number: int = 1
    total_emails: int = 1
    last_reward: Optional[float] = None
    done: bool = False
    feedback: str = ""

class EmailTriageState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    task_name: str = "easy"
    total_score: float = 0.0
    max_score: float = 1.0
