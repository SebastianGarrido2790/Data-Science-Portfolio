from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_importances(model: Any) -> tuple[list[str], list[float]] | tuple[None, None]:
    """
    Extracts feature importances or coefficients from an arbitrary trained machine learning model.

    Why no sklearn import?
    This function uses "duck typing" to inspect the model's attributes (`hasattr`) rather than
    relying on strict type checking (`isinstance()`) against specific libraries like scikit-learn
    or XGBoost. This approach reduces hard dependencies, prevents import errors if packages
    change, and makes the utility natively extendable across entirely different ML families.

    Args:
        model (Any): The trained machine learning model object (e.g., sklearn estimator).

    Returns:
        tuple[list[str], list[float]] | tuple[None, None]: A tuple containing two lists:
            - feature_names (list[str]): The names of the features.
            - importances (list[float]): The importance scores or absolute coefficients.
            Returns (None, None) if the model format is unsupported or an error occurs.
    """
    try:
        # Scikit-learn Tree-based models (e.g., RandomForest, GradientBoosting)
        if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
            logger.info("Extracting feature importances from tree-based model.")
            return list(model.feature_names_in_), list(model.feature_importances_)

        # Scikit-learn Linear models (e.g., Ridge, Lasso, ElasticNet)
        elif hasattr(model, "coef_") and hasattr(model, "feature_names_in_"):
            logger.info("Extracting feature importances from linear model coefficients.")
            # For linear models, the absolute magnitude of coefficients determines importance
            return list(model.feature_names_in_), list(np.abs(model.coef_))

        # Direct XGBoost Booster object extraction
        elif hasattr(model, "get_booster"):
            logger.info("Extracting feature importances via XGBoost booster gain.")
            booster = model.get_booster()
            importance_map = booster.get_score(importance_type="gain")
            return list(importance_map.keys()), list(importance_map.values())

        else:
            logger.warning(
                f"Feature importance extraction not supported for {type(model).__name__}."
            )
            return None, None

    except Exception as e:
        logger.error(f"Error occurred while extracting feature importances: {e}")
        return None, None
