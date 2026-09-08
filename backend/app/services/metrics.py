"""
Engagement metric formulas. Each function implements exactly one metric
so it can be unit tested and cited independently in the README's
Formula / Explanation / Interpretation table.
"""


def teacher_dominance_ratio(teacher_time: float, total_talk_time: float) -> float:
    """
    Formula: teacher_talk_time / total_talk_time
    Explanation: share of all speech that came from the teacher.
    Interpretation: >0.7 lecture-heavy session, <0.4 discussion-heavy session.
    """
    if total_talk_time <= 0:
        return 0.0
    return teacher_time / total_talk_time


def student_participation_indicator(student_turns: int, total_turns: int) -> float:
    """
    Formula: student_turns / total_turns
    Explanation: share of all speaking turns taken by students.
    Interpretation: higher values indicate participation spread across
    students rather than a teacher-dominated monologue.
    """
    if total_turns <= 0:
        return 0.0
    return student_turns / total_turns


def interaction_density(
    teacher_question_count: int,
    student_response_count: int,
    session_duration_seconds: float,
) -> float:
    """
    Formula: (teacher_questions + student_responses) / session_duration_minutes
    Explanation: rate of question-and-response interactions per minute.
    Interpretation: higher values suggest a more active, back-and-forth
    session rather than a passive one-directional lecture.
    """
    duration_minutes = session_duration_seconds / 60.0
    if duration_minutes <= 0:
        return 0.0
    return (teacher_question_count + student_response_count) / duration_minutes
