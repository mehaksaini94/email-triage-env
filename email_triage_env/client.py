
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import EmailTriageAction, EmailTriageObservation, EmailTriageState


class EmailTriageEnv(EnvClient):

    def _step_payload(self, action: EmailTriageAction) -> dict:
        return {
            "urgency": action.urgency,
            "category": action.category,
            "tone": action.tone,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        return StepResult(
            observation=EmailTriageObservation(
                email_subject=payload.get("email_subject", ""),
                email_body=payload.get("email_body", ""),
                task_name=payload.get("task_name", "easy"),
                step_number=payload.get("step_number", 1),
                total_emails=payload.get("total_emails", 1),
                last_reward=payload.get("last_reward"),
                done=payload.get("done", False),
                feedback=payload.get("feedback", "")
            ),
            reward=payload.get("last_reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> EmailTriageState:
        return EmailTriageState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", "easy"),
            total_score=payload.get("total_score", 0.0),
            max_score=payload.get("max_score", 1.0)
        )
