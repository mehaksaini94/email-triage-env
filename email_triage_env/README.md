
# Email Triage Environment

An OpenEnv reinforcement learning environment where AI agents learn
to triage customer support emails by classifying urgency, category,
and reply tone.

## Real-World Task

Companies receive hundreds of support emails daily. Correctly
prioritizing and routing them saves time and improves customer
satisfaction. This environment trains agents to do exactly that.

## Action Space

| Field    | Values                                        |
|----------|-----------------------------------------------|
| urgency  | urgent, not_urgent                            |
| category | billing, technical, general, complaint        |
| tone     | formal, empathetic, urgent                    |

## Observation Space

| Field         | Type    | Description                        |
|---------------|---------|------------------------------------|
| email_subject | string  | The email subject line             |
| email_body    | string  | The full email content             |
| task_name     | string  | Current task: easy/medium/hard     |
| step_number   | integer | Which email we are on              |
| total_emails  | integer | Total emails in this episode       |
| last_reward   | float   | Reward from previous action        |
| done          | boolean | Whether episode is complete        |
| feedback      | string  | What was correct or incorrect      |

## Tasks

| Task   | Emails | Difficulty | Description                          |
|--------|--------|------------|--------------------------------------|
| easy   | 3      | Easy       | Clear signals, obvious classification|
| medium | 5      | Medium     | Mixed signals, some ambiguity        |
| hard   | 10     | Hard       | Subtle signals, complex cases        |

## Reward Function

Per email: urgency correct (+0.4) + category correct (+0.4) + tone correct (+0.2) = 1.0 max
Episode score = average reward across all emails (0.0 to 1.0)

## Baseline Scores

| Task   | Random Agent | Smart Agent |
|--------|-------------|-------------|
| easy   | ~0.20       | ~0.80       |
| medium | ~0.20       | ~0.70       |
| hard   | ~0.20       | ~0.65       |

## Setup

```bash
pip install openenv-core
docker build -t email-triage-env ./server
docker run -p 8000:8000 email-triage-env
```

## Usage

```python
from email_triage_env.client import EmailTriageEnv
from email_triage_env.models import EmailTriageAction

async with EmailTriageEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="easy")
    result = await env.step(EmailTriageAction(
        urgency="urgent",
        category="technical",
        tone="formal"
    ))
    print(result.reward)
```
