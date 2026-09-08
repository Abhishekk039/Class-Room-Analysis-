"""
Rule-based classroom summary. Deliberately template-driven rather than
an LLM call, to stay consistent with the offline-first requirement and
keep the output deterministic and testable.
"""
from app.models.schemas import EngagementMetrics


def generate_summary(metrics: EngagementMetrics) -> str:
    if metrics.teacher_dominance_ratio >= 0.7:
        style = "This was a teacher-led session with limited two-way exchange."
    elif metrics.teacher_dominance_ratio <= 0.4:
        style = "This was a discussion-heavy session with substantial student speech."
    else:
        style = "This session had a balanced mix of teacher explanation and student participation."

    if metrics.interaction_density_per_minute >= 2:
        engagement = "Questions and responses occurred frequently throughout the session."
    elif metrics.interaction_density_per_minute >= 0.5:
        engagement = "There was a moderate amount of question-and-response interaction."
    else:
        engagement = "Very few question-and-response exchanges were detected."

    return f"{style} {engagement}"
