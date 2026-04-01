"""End-to-end tests for the GST Invoice Compliance Checker environment."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import EnvironmentManager, ALL_TASK_IDS
from app.models import ActionType, AgentAction


def test_all_tasks_load():
    """Verify all 9 tasks load without errors."""
    env = EnvironmentManager()
    tasks = env.get_task_list()
    assert len(tasks) == 9, f"Expected 9 tasks, got {len(tasks)}"
    print(f"  [PASS] All {len(tasks)} tasks loaded")


def test_task_difficulties():
    """Verify correct difficulty distribution."""
    env = EnvironmentManager()
    tasks = env.get_task_list()
    difficulties = [t.difficulty.value for t in tasks]
    assert difficulties.count("easy") == 3
    assert difficulties.count("medium") == 3
    assert difficulties.count("hard") == 3
    print("  [PASS] Difficulty distribution: 3 easy, 3 medium, 3 hard")


def test_reset_and_state():
    """Test reset and state endpoints for each task."""
    env = EnvironmentManager()
    for task_id in ALL_TASK_IDS:
        state = env.reset(task_id)
        assert state.task_id == task_id
        assert len(state.invoices) > 0
        assert state.step_count == 0
        assert state.done is False

        state2 = env.state()
        assert state2.task_id == task_id
    print(f"  [PASS] Reset/state works for all {len(ALL_TASK_IDS)} tasks")


def test_flag_issue_action():
    """Test that flag_issue action works correctly."""
    env = EnvironmentManager()
    state = env.reset("easy_1")

    action = AgentAction(
        action=ActionType.FLAG_ISSUE,
        invoice_id="INV-E1-001",
        field="invoice_number",
        category="missing_field",
        severity="critical",
        description="Invoice number is missing",
    )

    response = env.step(action)
    assert response.state.step_count == 1
    assert len(response.state.findings) == 1
    assert response.done is False
    print("  [PASS] flag_issue action works")


def test_approve_action():
    """Test that approve action works."""
    env = EnvironmentManager()
    env.reset("hard_2")

    action = AgentAction(
        action=ActionType.APPROVE,
        invoice_id="INV-H2-001",
    )

    response = env.step(action)
    assert response.state.step_count == 1
    assert response.reward > 0  # INV-H2-001 is clean, so reward should be positive
    print("  [PASS] approve action works (clean invoice gives positive reward)")


def test_submit_report():
    """Test that submit_report ends the session and provides a score."""
    env = EnvironmentManager()
    env.reset("easy_1")

    # Flag a correct issue
    env.step(AgentAction(
        action=ActionType.FLAG_ISSUE,
        invoice_id="INV-E1-001",
        field="invoice_number",
        category="missing_field",
        severity="critical",
        description="Invoice number is missing",
    ))

    # Submit report
    response = env.step(AgentAction(action=ActionType.SUBMIT_REPORT))
    assert response.done is True
    assert response.state.score is not None
    assert 0.0 <= response.state.score <= 1.0
    print(f"  [PASS] submit_report works (score={response.state.score:.4f})")


def test_grader_perfect_score():
    """Test that finding all issues gives a high score."""
    env = EnvironmentManager()
    state = env.reset("easy_1")

    # easy_1 has missing: invoice_number, recipient_name, recipient_gstin
    for field in ["invoice_number", "recipient_name", "recipient_gstin"]:
        env.step(AgentAction(
            action=ActionType.FLAG_ISSUE,
            invoice_id="INV-E1-001",
            field=field,
            category="missing_field",
            severity="critical",
            description=f"{field} is missing",
        ))

    response = env.step(AgentAction(action=ActionType.SUBMIT_REPORT))
    assert response.state.score is not None
    assert response.state.score >= 0.5, f"Expected score >= 0.5, got {response.state.score}"
    print(f"  [PASS] Finding key issues gives good score ({response.state.score:.4f})")


def test_grader_zero_score():
    """Test that submitting with no findings gives score 0."""
    env = EnvironmentManager()
    env.reset("easy_1")

    response = env.step(AgentAction(action=ActionType.SUBMIT_REPORT))
    assert response.state.score == 0.0
    print("  [PASS] No findings gives score 0.0")


def test_max_steps_auto_grade():
    """Test that reaching max steps triggers auto-grading."""
    env = EnvironmentManager()
    state = env.reset("easy_1")
    max_steps = state.max_steps

    for i in range(max_steps):
        response = env.step(AgentAction(
            action=ActionType.FLAG_ISSUE,
            invoice_id="INV-E1-001",
            field=f"test_field_{i}",
            category="missing_field",
            severity="minor",
            description=f"Test finding {i}",
        ))
        if response.done:
            break

    assert response.done is True
    assert response.state.score is not None
    print(f"  [PASS] Max steps ({max_steps}) triggers auto-grade")


def test_done_session_no_more_steps():
    """Test that a done session returns 0 reward on further steps."""
    env = EnvironmentManager()
    env.reset("easy_1")
    env.step(AgentAction(action=ActionType.SUBMIT_REPORT))

    response = env.step(AgentAction(action=ActionType.SUBMIT_REPORT))
    assert response.done is True
    assert response.reward == 0.0
    print("  [PASS] Done session rejects further steps")


def test_ground_truth_exists_for_all_tasks():
    """Verify every task has at least 1 ground truth issue (except clean invoices in batch)."""
    from app.tasks.easy import get_easy_task
    from app.tasks.medium import get_medium_task
    from app.tasks.hard import get_hard_task

    for task_id in ALL_TASK_IDS:
        if task_id.startswith("easy_"):
            _, _, gt = get_easy_task(task_id)
        elif task_id.startswith("medium_"):
            _, _, gt = get_medium_task(task_id)
        else:
            _, _, gt = get_hard_task(task_id)

        assert len(gt) > 0, f"Task {task_id} has no ground truth issues"
        print(f"  [PASS] {task_id}: {len(gt)} ground truth issues")


def test_fastapi_app_loads():
    """Verify the FastAPI app initializes without errors."""
    from app.main import app
    assert app.title == "GST Invoice Compliance Checker — OpenEnv"
    routes = [r.path for r in app.routes]
    assert "/reset" in routes
    assert "/state" in routes
    assert "/step" in routes
    assert "/tasks" in routes
    assert "/info" in routes
    assert "/grade" in routes
    print("  [PASS] FastAPI app loads with all routes")


if __name__ == "__main__":
    print("=" * 60)
    print("Running GST Invoice Compliance Checker Tests")
    print("=" * 60)

    tests = [
        test_all_tasks_load,
        test_task_difficulties,
        test_reset_and_state,
        test_flag_issue_action,
        test_approve_action,
        test_submit_report,
        test_grader_perfect_score,
        test_grader_zero_score,
        test_max_steps_auto_grade,
        test_done_session_no_more_steps,
        test_ground_truth_exists_for_all_tasks,
        test_fastapi_app_loads,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
