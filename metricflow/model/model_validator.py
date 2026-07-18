from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import logging
import multiprocessing
from typing import List, Sequence

from metricflow.model.objects.user_configured_model import UserConfiguredModel
from metricflow.model.parsing.dir_to_model import ModelBuildResult
from metricflow.model.validations.agg_time_dimension import AggregationTimeDimensionRule
from metricflow.model.validations.data_sources import DataSourceTimeDimensionWarningsRule, DataSourceValidityWindowRule
from metricflow.model.validations.dimension_const import DimensionConsistencyRule
from metricflow.model.validations.element_const import ElementConsistencyRule
from metricflow.model.validations.identifiers import (
    IdentifierConfigRule,
    IdentifierConsistencyRule,
    NaturalIdentifierConfigurationRule,
    OnePrimaryIdentifierPerDataSourceRule,
)
from metricflow.model.validations.materializations import ValidMaterializationRule
from metricflow.model.validations.measures import (
    PercentileAggregationRule,
    CountAggregationExprRule,
    DataSourceMeasuresUniqueRule,
    MeasureConstraintAliasesRule,
    MetricMeasuresRule,
    MeasuresNonAdditiveDimensionRule,
)
from metricflow.model.validations.metrics import CumulativeMetricRule, DerivedMetricRule
from metricflow.model.validations.non_empty import NonEmptyRule
from metricflow.model.validations.reserved_keywords import ReservedKeywordsRule
from metricflow.model.validations.unique_valid_name import UniqueAndValidNameRule
from metricflow.model.validations.validator_helpers import (
    ModelValidationResults,
    ModelValidationRule,
    ModelValidationException,
)

logger = logging.getLogger(__name__)


class ModelValidator:
    """A Validator that acts on UserConfiguredModel"""

    DEFAULT_RULES = (
        PercentileAggregationRule(),
        DerivedMetricRule(),
        CountAggregationExprRule(),
        DataSourceMeasuresUniqueRule(),
        DataSourceTimeDimensionWarningsRule(),
        DataSourceValidityWindowRule(),
        DimensionConsistencyRule(),
        ElementConsistencyRule(),
        IdentifierConfigRule(),
        IdentifierConsistencyRule(),
        NaturalIdentifierConfigurationRule(),
        OnePrimaryIdentifierPerDataSourceRule(),
        MeasureConstraintAliasesRule(),
        MetricMeasuresRule(),
        CumulativeMetricRule(),
        NonEmptyRule(),
        UniqueAndValidNameRule(),
        ValidMaterializationRule(),
        AggregationTimeDimensionRule(),
        ReservedKeywordsRule(),
        MeasuresNonAdditiveDimensionRule(),
    )

    def __init__(self, rules: Sequence[ModelValidationRule] = DEFAULT_RULES, max_workers: int = 1) -> None:
        """Constructor.

        Args:
            rules: List of validation rules to run. Defaults to DEFAULT_RULES
            max_workers: sets the max number of rules to run against the model concurrently
        """

        # Raises an error if 'rules' is an empty sequence or None
        if not rules:
            raise ValueError("ModelValidator 'rules' must be a sequence with at least one ModelValidationRule.")

        self._rules = rules
        self._max_workers = max_workers

    def validate_model(self, model: UserConfiguredModel) -> ModelBuildResult:
        """Validate a model according to configured rules."""
        serialized_model = model.json()

        results: List[ModelValidationResults] = []

        if self._max_workers == 1:
            # A single-worker process pool provides no parallelism and can be
            # unsafe after threaded runtimes such as LanceDB have initialized.
            # Keep the serialized validation boundary, but execute the rules in
            # the current process.
            serialized_results = [
                validation_rule.validate_model_serialized_for_multiprocessing(serialized_model)
                for validation_rule in self._rules
            ]
        else:
            # Do not inherit the application's global multiprocessing policy.
            # ``spawn`` starts validators with a clean runtime on macOS, Linux,
            # and Windows instead of forking live threads and async runtimes.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=self._max_workers, mp_context=context) as executor:
                futures = [
                    executor.submit(validation_rule.validate_model_serialized_for_multiprocessing, serialized_model)
                    for validation_rule in self._rules
                ]
                serialized_results = [future.result() for future in as_completed(futures)]

        for serialized_result in serialized_results:
            results.append(ModelValidationResults.parse_raw(serialized_result))

        return ModelBuildResult(model=model, issues=ModelValidationResults.merge(results))

    def checked_validations(self, model: UserConfiguredModel) -> UserConfiguredModel:  # chTODO: remember checked_build
        """Similar to validate(), but throws an exception if validation fails."""
        model_copy = copy.deepcopy(model)
        build_result = self.validate_model(model_copy)

        if build_result.issues.has_blocking_issues:
            raise ModelValidationException(issues=tuple(build_result.issues.all_issues))

        return model
