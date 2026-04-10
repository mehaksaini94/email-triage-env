#!/usr/bin/env python3
import os
import json
import textwrap
import requests
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "https://mhk-s-email-triage-env.hf.space")
BENCHMARK    = "email-triage"
TASKS        = ["easy", "medium", "hard"]
MAX_STEPS    = {"easy": 3, "medium": 5, "hard": 10}

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r}", flush=True)

def env_reset(task_name):
    os.environ["TASK_NAME"] = task_name
    r = requests.post(f"{ENV_URL}/reset", json={}, timeout=60)
    return r.json()

def env_step(action_dict):
    r = requests.post(f"{ENV_URL}/step", json=action_dict, timeout=60)
    return r.json()

SYSTEM_PROMPT = """You are an email triage assistant.
Respond with ONLY a JSON object like this:
{"urgency": "urgent", "category": "technical", "tone": "formal"}

urgency: urgent OR not_urgent
category: billing OR technical OR general OR complaint
tone: formal OR empathetic OR urgent

No explanation. Only JSON."""

def get_llm_action(client, subject, body):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Subject: {subject}\n\nBody: {body}"},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        data = json.loads(resp.choices[0].message.content.strip())
        urgency  = data.get("urgency", "not_urgent")
        category = data.get("category", "general")
        tone     = data.get("tone", "formal")
        if urgency not in ["urgent","not_urgent"]: urgency = "not_urgent"
        if category not in ["billing","technical","general","complaint"]: category = "general"
        if tone not in ["formal","empathetic","urgent"]: tone = "formal"
        return {"urgency": urgency, "category": category, "tone": tone}
    except:
        return {"urgency": "not_urgent", "category": "general", "tone": "formal"}

def run_task(client, task_name):
    max_steps = MAX_STEPS[task_name]
    rewards   = []
    steps     = 0
    score     = 0.5
    success   = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_name)
        obs    = result.get("observation", {})

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break
            action = get_llm_action(client, obs.get("email_subject",""), obs.get("email_body",""))
            action_str = f"urgency={action['urgency']},category={action['category']},tone={action['tone']}"
            result = env_step(action)
            obs    = result.get("observation", {})
            reward = result.get("reward") or 0.0
            done   = result.get("done", False)
            steps  = step
            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done)
            if done:
                break

        raw   = sum(rewards) / max_steps if max_steps > 0 else 0.5
        # STRICTLY between 0 and 1 — never 0.0, never 1.0
        score = round(min(max(raw, 0.01), 0.99), 3)
        success = score >= 0.5

    except Exception as e:
        log_step(step=steps+1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return score

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_task(client, task)
        print("", flush=True)

if __name__ == "__main__":
    main()
