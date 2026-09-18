import os
import json
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from sklearn.ensemble import HistGradientBoostingClassifier
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

RANDOM_SEED = 42

class BaseSyntheticGenerator(ABC):
    """Abstract base class for SYNTHIA clinical synthetic data generators."""
    
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def sample(self, num_rows: int = 1000, targets: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        pass


class HurdleConditionalCopulaModel(BaseSyntheticGenerator):
    """
    Primary Generative Model combining:
    1. Hurdle / Zero-Inflated Model for pain_score (matching 69.5% exact zero point mass)
    2. Class-Conditional Gaussian Copula for diabetes status (9.95% baseline prevalence or user-requested prevalence)
    3. Physiological and clinical bounds enforcement
    """
    def __init__(self, random_seed=RANDOM_SEED, default_distribution=None):
        self.random_seed = random_seed
        self.default_distribution = default_distribution
        self.hurdle_clf = None
        self.copula_diab0 = None
        self.copula_diab1 = None
        self.diab_prevalence = 0.0995
        self.pain_zero_prop = 0.695
        self.cols = ['age', 'sex', 'diabetes', 'systolic_bp', 'diastolic_bp', 'activity_mims', 'n_medications', 'adherence_pct', 'pain_score']
        self.other_cols_for_pain = ['age', 'sex', 'diabetes', 'systolic_bp', 'diastolic_bp', 'activity_mims', 'n_medications', 'adherence_pct']
        self.model_name = "Hurdle Conditional Copula Model"
        self.model_type = "HurdleConditionalCopulaSynthesizer"
        self.provenance = "Trained on 4,826 NHANES Cycle G clinical records"

    def fit(self, train_df: pd.DataFrame):
        np.random.seed(self.random_seed)
        
        # 1. Fit Pain Hurdle Zero/Nonzero Classifier
        y_pain_zero = (train_df['pain_score'] == 0.0).astype(int)
        self.pain_zero_prop = float(y_pain_zero.mean())
        
        X_pain_feat = train_df[self.other_cols_for_pain]
        self.hurdle_clf = HistGradientBoostingClassifier(random_state=self.random_seed, max_iter=200)
        self.hurdle_clf.fit(X_pain_feat, y_pain_zero)
        
        # 2. Fit Class-Conditional Copulas for diabetes = 0 vs diabetes = 1
        df_diab0 = train_df[train_df['diabetes'] == 0].copy()
        df_diab1 = train_df[train_df['diabetes'] == 1].copy()
        
        self.diab_prevalence = float((train_df['diabetes'] == 1).mean())
        
        meta0 = Metadata.detect_from_dataframe(data=df_diab0, table_name='nhanes_d0')
        meta0.update_column(column_name='sex', sdtype='categorical')
        meta0.update_column(column_name='diabetes', sdtype='categorical')
        
        meta1 = Metadata.detect_from_dataframe(data=df_diab1, table_name='nhanes_d1')
        meta1.update_column(column_name='sex', sdtype='categorical')
        meta1.update_column(column_name='diabetes', sdtype='categorical')
        
        self.copula_diab0 = GaussianCopulaSynthesizer(meta0, enforce_min_max_values=True)
        self.copula_diab0.fit(df_diab0)
        
        self.copula_diab1 = GaussianCopulaSynthesizer(meta1, enforce_min_max_values=True)
        self.copula_diab1.fit(df_diab1)

    def sample(self, num_rows: int = 1000, targets: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        np.random.seed(self.random_seed)
        targets = targets or {}
        
        # Determine target diabetes prevalence
        if "diabetes" in targets:
            diab_val = targets["diabetes"]
            # Check if percentage (e.g. 30 for 30%) or binary (1)
            if diab_val > 1.0:
                p_diab = float(diab_val) / 100.0
            elif diab_val == 1 or diab_val == 1.0:
                p_diab = 1.0
            elif diab_val == 0 or diab_val == 0.0:
                p_diab = 0.0
            else:
                p_diab = float(diab_val)
        else:
            p_diab = self.diab_prevalence

        p_diab = max(0.0, min(1.0, p_diab))
        n1 = int(round(num_rows * p_diab))
        n0 = num_rows - n1
        
        sampled_parts = []
        if n0 > 0:
            syn0 = self.copula_diab0.sample(num_rows=n0)
            syn0['diabetes'] = 0
            sampled_parts.append(syn0)
        if n1 > 0:
            syn1 = self.copula_diab1.sample(num_rows=n1)
            syn1['diabetes'] = 1
            sampled_parts.append(syn1)
            
        if sampled_parts:
            combined_syn = pd.concat(sampled_parts, ignore_index=True)
        else:
            combined_syn = self.copula_diab0.sample(num_rows=num_rows)
            combined_syn['diabetes'] = 0
            
        combined_syn = combined_syn.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        combined_syn = combined_syn[self.cols]
        
        # Apply Pain Hurdle Zero/Nonzero Model
        X_pain = combined_syn[self.other_cols_for_pain]
        p_zero = self.hurdle_clf.predict_proba(X_pain)[:, 1]  # Prob(pain == 0)
        
        is_zero = np.random.binomial(1, p_zero).astype(bool)
        combined_syn.loc[is_zero, 'pain_score'] = 0.0
        
        non_zero_mask = ~is_zero
        combined_syn.loc[non_zero_mask, 'pain_score'] = combined_syn.loc[non_zero_mask, 'pain_score'].clip(lower=0.1, upper=10.0).round(1)
        
        # Target conditioning adjustments for continuous features
        # 1. Age adjustments
        if "age_gt" in targets or "ageOver60" in targets:
            target_age_pct = targets.get("ageOver60")
            if target_age_pct is None and "age_gt" in targets:
                # If target is age_gt 60, target 100% or user specified
                target_age_pct = 100.0 if targets.get("age_gt") == 60 else None
            if target_age_pct is not None:
                p_age = float(target_age_pct) / 100.0 if target_age_pct > 1.0 else float(target_age_pct)
                cur_pct = (combined_syn['age'] >= 60).mean()
                diff = p_age - cur_pct
                if abs(diff) > 0.03:
                    shift = diff * 22.0
                    combined_syn['age'] = combined_syn['age'] + shift

        # 2. Activity adjustments
        if "lowActivity" in targets or "activity_mims_lt" in targets:
            target_act_pct = targets.get("lowActivity")
            if target_act_pct is not None:
                p_low = float(target_act_pct) / 100.0 if target_act_pct > 1.0 else float(target_act_pct)
                cur_low = (combined_syn['activity_mims'] < 8500).mean()
                diff = p_low - cur_low
                if abs(diff) > 0.03:
                    shift = -diff * 5000.0
                    combined_syn['activity_mims'] = combined_syn['activity_mims'] + shift

        # General physiological constraint enforcement
        combined_syn['age'] = combined_syn['age'].clip(8, 85).round().astype(int)
        combined_syn['sex'] = combined_syn['sex'].round().clip(1, 2).astype(int)
        combined_syn['diabetes'] = combined_syn['diabetes'].round().clip(0, 1).astype(int)
        combined_syn['systolic_bp'] = combined_syn['systolic_bp'].clip(70, 233).round().astype(int)
        combined_syn['diastolic_bp'] = combined_syn['diastolic_bp'].clip(14, 116).round().astype(int)
        # Ensure DBP < SBP - 10
        combined_syn['diastolic_bp'] = np.minimum(combined_syn['diastolic_bp'], combined_syn['systolic_bp'] - 10)
        combined_syn['activity_mims'] = combined_syn['activity_mims'].clip(0, 35000).round().astype(int)
        combined_syn['n_medications'] = combined_syn['n_medications'].clip(0, 19).round().astype(int)
        combined_syn['adherence_pct'] = combined_syn['adherence_pct'].clip(0.0, 100.0).round(1)
        combined_syn['pain_score'] = combined_syn['pain_score'].clip(0.0, 10.0).round(1)
        
        return combined_syn
