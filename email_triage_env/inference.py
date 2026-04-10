
import os
import json
import asyncio
import textwrap
from typing import List, Optional
from openai import OpenAI
from client import EmailTriageEnv
from models import EmailTriageAction

# ─────────────────────────────────────────────
# CONFIG — reads from environment variables
# ─────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK    = "email-triage"

TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = {"easy": 3, "medium": 5, "hard": 10}


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val  = str(done).lower()
    reward_val = f"{reward:.2f}" if reward is not None else "0.00"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward_val} done={done_val} error={error_val}",
        flush=True
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# ─────────────────────────────────────────────
# SYSTEM PROMPT — tells LLM what to do
# ─────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an email triage assistant.
    You will be shown a support email subject and body.
    You must classify it by responding with ONLY a JSON object like this:

    {
        "urgency": "urgent",
        "category": "technical",
        "tone": "formal"
    }

    Rules:
    - urgency must be exactly: urgent OR not_urgent
    - category must be exactly: billing OR technical OR general OR complaint
    - tone must be exactly: formal OR empathetic OR urgent

    Urgency guide:
    - urgent: needs same-day response, system down, angry customer, security issue
    - not_urgent: general questions, feature requests, feedback

    Category guide:
    - billing: payments, invoices, refunds, charges
    - technical: bugs, errors, login issues, API, integrations
    - complaint: unhappy customer, threatening to leave, bad experience
    - general: questions, feedback, partnerships, feature requests

    Tone guide:
    - urgent: system is down, security breach, immediate action needed
    - empathetic: customer is upset, complaint, emotional situation
    - formal: neutral professional tone for general queries

    Respond with ONLY the JSON. No explanation. No extra text.
""").strip()

# ─────────────────────────────────────────────
# GET LLM DECISION
# ─────────────────────────────────────────────
def get_llm_action(client: OpenAI, subject: str, body: str) -> EmailTriageAction:
    user_prompt = f"Subject: {subject}\n\nBody: {body}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()

        # parse JSON response
        data = json.loads(text)
        urgency  = data.get("urgency",  "not_urgent")
        category = data.get("category", "general")
        tone     = data.get("tone",     "formal")

        # validate values
        if urgency  not in ["urgent", "not_urgent"]:
            urgency = "not_urgent"
        if category not in ["billing", "technical", "general", "complaint"]:
            category = "general"
        if tone not in ["formal", "empathetic", "urgent"]:
            tone = "formal"

        return EmailTriageAction(
            urgency=urgency,
            category=category,
            tone=tone
        )

    except Exception as e:
        # fallback if LLM fails
        return EmailTriageAction(
            urgency="not_urgent",
            category="general",
            tone="formal"
        )

# ─────────────────────────────────────────────
# RUN ONE TASK
# ─────────────────────────────────────────────
async def run_task(client: OpenAI, env: EmailTriageEnv, task_name: str):
    max_steps = MAX_STEPS_PER_TASK[task_name]
    rewards   = []
    steps     = 0
    score     = 0.0
    success   = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # reset environment for this task
        result = await env.reset(task=task_name)
        obs    = result.observation

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            # LLM decides action
            action = get_llm_action(
                client,
                obs.email_subject,
                obs.email_body
            )

            action_str = (
                f"urgency={action.urgency},"
                f"category={action.category},"
                f"tone={action.tone}"
            )

            # send action to environment
            result  = await env.step(action)
            obs     = result.observation
            reward  = result.reward or 0.0
            done    = result.done
            steps   = step

            rewards.append(reward)
            log_step(step=step, action=action_str,
                     reward=reward, done=done)

            if done:
                break

        # calculate final score
        score   = sum(rewards) / max_steps if max_steps > 0 else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= 0.5

    except Exception as e:
        log_step(step=steps+1, action="error",
                 reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps,
                score=score, rewards=rewards)

    return score

# ─────────────────────────────────────────────
# MAIN — runs all 3 tasks
# ─────────────────────────────────────────────
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with EmailTriageEnv(base_url=ENV_URL) as env:
        for task in TASKS:
            await run_task(client, env, task)
            print("", flush=True)  # blank line between tasks

if __name__ == "__main__":
    asyncio.run(main())
