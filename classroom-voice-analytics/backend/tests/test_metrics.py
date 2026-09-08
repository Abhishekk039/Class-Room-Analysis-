from app.services.metrics import (
    teacher_dominance_ratio,
    student_participation_indicator,
    interaction_density,
)


def test_teacher_dominance_ratio_basic():
    assert teacher_dominance_ratio(70, 100) == 0.7


def test_teacher_dominance_ratio_zero_total():
    assert teacher_dominance_ratio(0, 0) == 0.0


def test_student_participation_indicator_basic():
    assert student_participation_indicator(4, 10) == 0.4


def test_student_participation_indicator_zero_turns():
    assert student_participation_indicator(0, 0) == 0.0


def test_interaction_density_basic():
    # 6 questions + 4 responses over 5 minutes = 2.0 per minute
    assert interaction_density(6, 4, 300) == 2.0


def test_interaction_density_zero_duration():
    assert interaction_density(5, 5, 0) == 0.0
