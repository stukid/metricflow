import pytest
from typing import Any, Callable

from metricflow.dataset.dataset import DataSet
from metricflow.model.model_validator import ModelValidator
from metricflow.model.validations.non_empty import NonEmptyRule
from metricflow.model.validations.materializations import ValidMaterializationRule
from metricflow.model.objects.user_configured_model import UserConfiguredModel
from metricflow.model import model_validator as model_validator_module
from metricflow.test.model.validations.helpers import materialization_with_guaranteed_meta
from metricflow.test.test_utils import model_with_materialization


def test_can_configure_model_validator_rules(  # noqa: D
    simple_model__with_primary_transforms: UserConfiguredModel,
) -> None:
    model = model_with_materialization(
        simple_model__with_primary_transforms,
        [
            materialization_with_guaranteed_meta(
                name="foobar",
                metrics=["invalid_bookings"],
                dimensions=[DataSet.metric_time_dimension_name()],
            )
        ],
    )

    # confirm that with the default configuration, an issue is raised
    issues = ModelValidator().validate_model(model).issues
    assert len(issues.all_issues) == 1, f"ModelValidator with default rules had unexpected number of issues {issues}"

    # confirm that a custom configuration excluding ValidMaterializationRule, no issue is raised
    rules = [rule for rule in ModelValidator.DEFAULT_RULES if rule.__class__ is not ValidMaterializationRule]
    issues = ModelValidator(rules=rules).validate_model(model).issues
    assert len(issues.all_issues) == 0, f"ModelValidator without ValidMaterializationRule returned issues {issues}"


def test_cant_configure_model_validator_without_rules() -> None:  # noqa: D
    with pytest.raises(ValueError):
        ModelValidator(rules=[])

    with pytest.raises(ValueError):
        ModelValidator(rules=())

    with pytest.raises(ValueError):
        ModelValidator(rules=None)  # type: ignore


def test_model_validator_releases_process_pool(
    simple_model__with_primary_transforms: UserConfiguredModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ImmediateFuture:
        def __init__(self, value: str) -> None:
            self._value = value

        def result(self) -> str:
            return self._value

    class RecordingExecutor:
        instances: list["RecordingExecutor"] = []

        def __init__(self, max_workers: int, mp_context: Any) -> None:
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.entered = False
            self.exited = False
            RecordingExecutor.instances.append(self)

        def __enter__(self) -> "RecordingExecutor":
            self.entered = True
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            self.exited = True

        def submit(self, fn: Callable[..., str], *args: Any) -> ImmediateFuture:
            return ImmediateFuture(fn(*args))

    monkeypatch.setattr(model_validator_module, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(model_validator_module, "as_completed", lambda futures: futures)

    result = ModelValidator(rules=[NonEmptyRule()], max_workers=7).validate_model(simple_model__with_primary_transforms)

    assert result.issues.all_issues == ()
    assert len(RecordingExecutor.instances) == 1
    executor = RecordingExecutor.instances[0]
    assert executor.max_workers == 7
    assert executor.mp_context.get_start_method() == "spawn"
    assert executor.entered is True
    assert executor.exited is True


def test_model_validator_runs_single_worker_in_current_process(
    simple_model__with_primary_transforms: UserConfiguredModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_process_pool(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("single-worker validation should not create a process pool")

    monkeypatch.setattr(model_validator_module, "ProcessPoolExecutor", unexpected_process_pool)

    result = ModelValidator(rules=[NonEmptyRule()], max_workers=1).validate_model(simple_model__with_primary_transforms)

    assert result.issues.all_issues == ()


def test_model_validator_runs_parallel_rules_with_spawn(
    simple_model__with_primary_transforms: UserConfiguredModel,
) -> None:
    result = ModelValidator(rules=[NonEmptyRule()], max_workers=2).validate_model(simple_model__with_primary_transforms)

    assert result.issues.all_issues == ()
