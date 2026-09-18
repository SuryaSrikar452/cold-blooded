import os
import time
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from backend.services.hurdle_copula_model import HurdleConditionalCopulaModel, BaseSyntheticGenerator

logger = logging.getLogger("synthia.generator")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = BASE_DIR / "final_training_data" / "nhanes_generative_train.csv"
MODEL_CACHE_DIR = BASE_DIR / "backend" / "models" / "hurdle_copula"
MODEL_CACHE_FILE = MODEL_CACHE_DIR / "hurdle_model.pkl"

# In-memory singletons
_ACTIVE_GENERATOR: Optional[BaseSyntheticGenerator] = None
_COHORT_STORE: Dict[str, Dict[str, Any]] = {}
_LATEST_COHORT_ID: Optional[str] = None


def get_generator() -> HurdleConditionalCopulaModel:
    """
    Returns the cached HurdleConditionalCopulaModel instance.
    Loads from serialized cache if present, otherwise trains on startup and caches in memory.
    Raises RuntimeError if train data is missing or model cannot initialize (zero silent fallbacks).
    """
    global _ACTIVE_GENERATOR
    if _ACTIVE_GENERATOR is not None:
        return _ACTIVE_GENERATOR

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try loading from disk cache
    if MODEL_CACHE_FILE.exists():
        try:
            logger.info(f"Loading cached Hurdle Conditional Copula Model from {MODEL_CACHE_FILE}...")
            with open(MODEL_CACHE_FILE, "rb") as f:
                _ACTIVE_GENERATOR = pickle.load(f)
            logger.info("Cached Hurdle Model loaded successfully.")
            return _ACTIVE_GENERATOR
        except Exception as e:
            logger.warning(f"Failed to load cached model artifact: {e}. Retraining...")

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Generative training dataset not found at {TRAIN_CSV}. Cannot initialize generator.")

    logger.info("Fitting Hurdle Conditional Copula Model on training corpus...")
    t0 = time.time()
    train_df = pd.read_csv(TRAIN_CSV)
    model = HurdleConditionalCopulaModel(random_seed=42)
    model.fit(train_df)
    t_fit = time.time() - t0
    logger.info(f"Hurdle Model trained successfully in {t_fit:.2f}s.")

    # Save artifact for faster restarts
    try:
        with open(MODEL_CACHE_FILE, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved trained Hurdle model artifact to {MODEL_CACHE_FILE}.")
    except Exception as save_err:
        logger.warning(f"Could not serialize model to disk: {save_err}")

    _ACTIVE_GENERATOR = model
    return _ACTIVE_GENERATOR


def store_generated_cohort(cohort_id: str, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
    """Stores generated cohort in the in-memory store and updates latest cohort reference."""
    global _LATEST_COHORT_ID
    _COHORT_STORE[cohort_id] = {
        "df": df.copy(),
        "metadata": metadata or {},
        "created_at": time.time()
    }
    _LATEST_COHORT_ID = cohort_id


def get_cohort_dataframe(cohort_id: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieves the generated synthetic cohort dataframe by cohort_id, or returns the latest generated cohort.
    If no cohort has been generated yet in this session, raises an error or loads the initial benchmark sample.
    """
    global _LATEST_COHORT_ID
    target_id = cohort_id or _LATEST_COHORT_ID
    
    if target_id and target_id in _COHORT_STORE:
        return _COHORT_STORE[target_id]["df"].copy()

    # Check disk storage for previous cohort run
    cohorts_dir = BASE_DIR / "backend" / "generated_cohorts"
    if target_id:
        csv_file = cohorts_dir / f"{target_id}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            _COHORT_STORE[target_id] = {"df": df, "metadata": {}, "created_at": time.time()}
            _LATEST_COHORT_ID = target_id
            return df

    # If any cohort exists in cohorts_dir, load latest
    if cohorts_dir.exists():
        csvs = sorted(cohorts_dir.glob("*.csv"), key=os.path.getmtime, reverse=True)
        if csvs:
            latest_csv = csvs[0]
            df = pd.read_csv(latest_csv)
            cid = latest_csv.stem
            _COHORT_STORE[cid] = {"df": df, "metadata": {}, "created_at": time.time()}
            _LATEST_COHORT_ID = cid
            return df

    # If no cohort generated yet, generate a live default baseline cohort with the Hurdle model!
    logger.info("No active cohort found in session. Generating initial live baseline cohort via Hurdle model...")
    gen = get_generator()
    default_df = gen.sample(num_rows=1000)
    cid = f"SYN-HC-INIT{int(time.time())}"
    store_generated_cohort(cid, default_df, {"initial_baseline": True})
    return default_df


def get_latest_cohort_id() -> Optional[str]:
    return _LATEST_COHORT_ID
