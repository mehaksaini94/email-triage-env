---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

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

| Task   | Emails | Description                           |
|--------|--------|---------------------------------------|
| easy   | 3      | Clear signals, obvious classification |
| medium | 5      | Mixed signals, some ambiguity         |
| hard   | 10     | Subtle signals, complex cases         |

## Reward Function

- urgency correct: +0.4
- category correct: +0.4
- tone correct: +0.2
- Total per email: 1.0 max
- Episode score = average reward (0.0 to 1.0)

## API Endpoints

- POST /reset — start new episode
- POST /step — take action
- GET /state — get metadata
- GET /health — health check
- GET /tasks — list all tasks
- GET /grader — get episode score
