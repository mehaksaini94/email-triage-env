import uuid
import os

EMAILS = {
    "easy": [
        {"subject": "URGENT: Server is down!", "body": "Our production server crashed 10 minutes ago. All services are offline. We need immediate help!", "correct_urgency": "urgent", "correct_category": "technical", "correct_tone": "urgent"},
        {"subject": "Question about my invoice", "body": "Hi, I received my invoice for this month but the amount seems different. Could you explain?", "correct_urgency": "not_urgent", "correct_category": "billing", "correct_tone": "formal"},
        {"subject": "General enquiry", "body": "Hello, I wanted to know more about your services and pricing options. No rush.", "correct_urgency": "not_urgent", "correct_category": "general", "correct_tone": "formal"},
    ],
    "medium": [
        {"subject": "Not happy with service", "body": "I have been a customer for 3 years and this is the worst experience. My issue has not been resolved for 2 weeks.", "correct_urgency": "urgent", "correct_category": "complaint", "correct_tone": "empathetic"},
        {"subject": "Payment failed twice", "body": "My payment has failed twice this week. I tried a different card but still no luck.", "correct_urgency": "urgent", "correct_category": "billing", "correct_tone": "empathetic"},
        {"subject": "How do I reset my password?", "body": "I forgot my password and cannot log in. The reset email is not arriving.", "correct_urgency": "not_urgent", "correct_category": "technical", "correct_tone": "formal"},
        {"subject": "Feedback on new feature", "body": "I just tried your new dashboard. It looks great but the export button is confusing.", "correct_urgency": "not_urgent", "correct_category": "general", "correct_tone": "formal"},
        {"subject": "Threatening to cancel", "body": "If this is not fixed by tomorrow I will cancel my subscription and leave a public review.", "correct_urgency": "urgent", "correct_category": "complaint", "correct_tone": "empathetic"},
    ],
    "hard": [
        {"subject": "Follow up on last week", "body": "The integration is still not working. This is affecting our daily operations significantly.", "correct_urgency": "urgent", "correct_category": "technical", "correct_tone": "empathetic"},
        {"subject": "Billing discrepancy", "body": "We were charged for enterprise but only signed up for basic. This has happened three months in a row.", "correct_urgency": "urgent", "correct_category": "billing", "correct_tone": "empathetic"},
        {"subject": "API rate limits", "body": "We are hitting rate limits during peak hours. We need higher limits or optimization advice.", "correct_urgency": "not_urgent", "correct_category": "technical", "correct_tone": "formal"},
        {"subject": "Partnership opportunity", "body": "Our company would like to explore a potential partnership. We believe there is mutual value.", "correct_urgency": "not_urgent", "correct_category": "general", "correct_tone": "formal"},
        {"subject": "Data breach concern", "body": "I think my account may have been compromised. I noticed logins from unknown locations.", "correct_urgency": "urgent", "correct_category": "technical", "correct_tone": "urgent"},
        {"subject": "Refund not received", "body": "I requested a refund 3 weeks ago and still have not received it.", "correct_urgency": "urgent", "correct_category": "billing", "correct_tone": "empathetic"},
        {"subject": "Feature request", "body": "It would be great if you could add dark mode to the mobile app.", "correct_urgency": "not_urgent", "correct_category": "general", "correct_tone": "formal"},
        {"subject": "Extremely disappointed", "body": "I have contacted support four times with no resolution. I want a manager.", "correct_urgency": "urgent", "correct_category": "complaint", "correct_tone": "empathetic"},
        {"subject": "Wrong item delivered", "body": "My order arrived but it was the wrong item. I need the correct item before Friday.", "correct_urgency": "urgent", "correct_category": "complaint", "correct_tone": "urgent"},
        {"subject": "How to export data?", "body": "Could you tell me how to export my data in CSV format? I checked the docs but could not find it.", "correct_urgency": "not_urgent", "correct_category": "general", "correct_tone": "formal"},
    ]
}


class EmailTriageEnvironment:
    def __init__(self):
        self.task_name = "easy"
        self.emails = []
        self.current_index = 0
        self.episode_id = ""
        self.step_count = 0
        self.total_score = 0.0
        self.done = False

    def reset(self):
        task_name = os.getenv("TASK_NAME", "easy")
        self.task_name = task_name
        self.emails = list(EMAILS[task_name])
        self.current_index = 0
        self.step_count = 0
        self.total_score = 0.0
        self.done = False
        self.episode_id = str(uuid.uuid4())
        email = self.emails[0]
        return {
            "email_subject": email["subject"],
            "email_body": email["body"],
            "task_name": self.task_name,
            "step_number": 1,
            "total_emails": len(self.emails),
            "last_reward": None,
            "done": False,
            "feedback": "New episode started. Triage this email.",
        }

    def step(self, action):
        if self.done:
            return {"email_subject": "", "email_body": "", "task_name": self.task_name,
                    "step_number": self.step_count, "total_emails": len(self.emails),
                    "last_reward": 0.0, "done": True, "feedback": "Episode already done."}

        email = self.emails[self.current_index]
        if isinstance(action, dict):
            urgency = action.get("urgency", "")
            category = action.get("category", "")
            tone = action.get("tone", "")
        else:
            urgency = getattr(action, "urgency", "")
            category = getattr(action, "category", "")
            tone = getattr(action, "tone", "")

        reward, feedback = self._grade(urgency, category, tone, email)
        self.total_score += reward
        self.step_count += 1
        self.current_index += 1

        if self.current_index >= len(self.emails):
            self.done = True
            return {
                "email_subject": "",
                "email_body": "",
                "task_name": self.task_name,
                "step_number": self.step_count,
                "total_emails": len(self.emails),
                "last_reward": reward,
                "done": True,
                "feedback": f"Episode complete! Score: {self.total_score:.2f}/{len(self.emails)}"
            }

        next_email = self.emails[self.current_index]
        return {
            "email_subject": next_email["subject"],
            "email_body": next_email["body"],
            "task_name": self.task_name,
            "step_number": self.step_count + 1,
            "total_emails": len(self.emails),
            "last_reward": reward,
            "done": False,
            "feedback": feedback
        }

    def _grade(self, urgency, category, tone, email):
        score = 0.0
        parts = []
        if urgency == email["correct_urgency"]:
            score += 0.4
            parts.append("urgency correct")
        else:
            parts.append(f"urgency wrong (expected {email['correct_urgency']})")
        if category == email["correct_category"]:
            score += 0.4
            parts.append("category correct")
        else:
            parts.append(f"category wrong (expected {email['correct_category']})")
        if tone == email["correct_tone"]:
            score += 0.2
            parts.append("tone correct")
        else:
            parts.append(f"tone wrong (expected {email['correct_tone']})")
        return round(score, 2), " | ".join(parts)

    def grader_score(self):
        if not self.emails:
            return 0.5
        raw = self.total_score / len(self.emails)
        # strictly between 0 and 1 — never exactly 0.0 or 1.0
        clamped = max(0.01, min(0.99, raw))
        return round(clamped, 3)

    @property
    def state(self):
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_name": self.task_name,
            "total_score": self.total_score,
            "max_score": float(len(self.emails))
        }
