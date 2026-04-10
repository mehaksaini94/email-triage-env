#!/usr/bin/env python3
"""
inference.py - Email Triage Environment
Connects directly to the HF Space URL using requests.
No local imports needed.
"""

import os
import json
import asyncio
import textwrap
import requests
from typing import List, Optional
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "https://mhk-s-email-triage-env.hf.space")
BENCHMARK    = "email-triage"

TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = {"easy": 3, "medium": 5, "hard": 10}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
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
# ENV CALLS — direct HTTP, no client.py needed
# ─────────────────────────────────────────────
def env_reset(task_name):
    os.environ["TASK_NAME"] = task_name
    r = requests.post(f"{ENV_URL}/reset", json={}, timeout=30)
    return r.json()

def env_step(action_dict):
    r = requests.post(f"{ENV_URL}/step", json=action_dict, timeout=30)
    return r.json()

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an email triage assistant.
    You will be shown a support email subject and body.
    Respond with ONLY a JSON object like this:

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
    - urgent: system down, security issue, angry customer, immediate action needed
    - not_urgent: general questions, feature requests, feedback

    Category guide:
    - billing: payments, invoices, refunds, charges
    - technical: bugs, errors, login issues, API problems
    - complaint: unhappy customer, threatening to leave
    - general: questions, feedback, partnerships

    Tone guide:
    - urgent: system down, security breach
    - empathetic: customer is upset or emotional
    - formal: neutral professional queries

    Respond with ONLY the JSON. No explanation.
""").strip()

# ─────────────────────────────────────────────
# GET LLM DECISION
# ─────────────────────────────────────────────
def get_llm_action(client, subject, body):
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
        data = json.loads(text)
        urgency  = data.get("urgency",  "not_urgent")
        category = data.get("category", "general")
        tone     = data.get("tone",     "formal")

        if urgency  not in ["urgent", "not_urgent"]:
            urgency = "not_urgent"
        if category not in ["billing", "technical", "general", "complaint"]:
            category = "general"
        if tone not in ["formal", "empathetic", "urgent"]:
            tone = "formal"

        return {"urgency": urgency, "category": category, "tone": tone}

    except Exception as e:
        return {"urgency": "not_urgent", "category": "general", "tone": "formal"}

# ─────────────────────────────────────────────
# RUN ONE TASK
# ─────────────────────────────────────────────
def run_task(client, task_name):
    max_steps = MAX_STEPS_PER_TASK[task_name]
    rewards   = []
    steps     = 0
    score     = 0.0
    success   = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_name)
        obs    = result.get("observation", {})

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            action = get_llm_action(
                client,
                obs.get("email_subject", ""),
                obs.get("email_body", "")
            )

            action_str = (
                f"urgency={action['urgency']},"
                f"category={action['category']},"
                f"tone={action['tone']}"
            )

            result  = env_step(action)
            obs     = result.get("observation", {})
            reward  = result.get("reward") or 0.0
            done    = result.get("done", False)
            steps   = step

            rewards.append(reward)
            log_step(step=step, action=action_str,
                     reward=reward, done=done)

            if done:
                break

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
# MAIN
# ─────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_task(client, task)
        print("", flush=True)

if __name__ == "__main__":
    main()
