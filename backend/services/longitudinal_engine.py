import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Literature-derived mean reversion parameters for clinical trajectories
# dx_t = theta * (mu - x_{t-1}) + sigma * epsilon_t
PROG_PARAMS = {
    'systolic_bp': {'theta': 0.15, 'sigma': 3.5, 'min': 70.0, 'max': 220.0},
    'diastolic_bp': {'theta': 0.15, 'sigma': 2.5, 'min': 40.0, 'max': 120.0},
    'activity_mims': {'theta': 0.20, 'sigma': 450.0, 'min': 0.0, 'max': 35000.0},
    'adherence_pct': {'theta': 0.10, 'sigma': 4.0, 'min': 0.0, 'max': 100.0},
    'pain_score': {'theta': 0.25, 'sigma': 0.6, 'min': 0.0, 'max': 10.0},
    'heart_rate': {'theta': 0.20, 'sigma': 3.0, 'min': 50.0, 'max': 120.0}
}

def generate_patient_trajectory(patient_dict, num_months=12, cadence="monthly", seed=RANDOM_SEED):
    """
    Generates an AR(1) mean-reverting longitudinal trajectory for a synthetic patient.
    Supports monthly (12 points, day_offset 0 to 360) or weekly cadence.
    """
    np.random.seed((seed + int(patient_dict.get('age', 40)) * 17) % 2**32)
    
    trajectory = []
    
    # Initialize baseline month 0 / day 0
    m0 = {
        'month': 0,
        'day_offset': 0,
        'age': float(patient_dict.get('age', 40)),
        'sex': int(patient_dict.get('sex', 1)),
        'diabetes': int(patient_dict.get('diabetes', 0)),
        'systolic_bp': float(patient_dict.get('systolic_bp', 120.0)),
        'diastolic_bp': float(patient_dict.get('diastolic_bp', 75.0)),
        'activity_mims': float(patient_dict.get('activity_mims', 10000.0)),
        'n_medications': int(patient_dict.get('n_medications', 1)),
        'adherence_pct': float(patient_dict.get('adherence_pct', 50.0)),
        'pain_score': float(patient_dict.get('pain_score', 1.0)),
        'heart_rate': float(patient_dict.get('heart_rate', 72.0 if patient_dict.get('systolic_bp', 120) < 140 else 78.0))
    }
    trajectory.append(m0)
    
    # Baseline setpoints (mean reversion targets)
    sbp_mu = m0['systolic_bp']
    dbp_mu = m0['diastolic_bp']
    act_mu = m0['activity_mims']
    adh_mu = m0['adherence_pct']
    pain_mu = m0['pain_score']
    hr_mu = m0['heart_rate']
    
    total_steps = num_months if cadence == "monthly" else num_months * 4
    step_days = 30 if cadence == "monthly" else 7

    for step in range(1, total_steps):
        prev = trajectory[-1]
        
        # SBP step
        p = PROG_PARAMS['systolic_bp']
        sbp_next = prev['systolic_bp'] + p['theta'] * (sbp_mu - prev['systolic_bp']) + np.random.normal(0, p['sigma'])
        sbp_next = np.clip(sbp_next, p['min'], p['max'])
        
        # DBP step (correlated with SBP)
        p = PROG_PARAMS['diastolic_bp']
        dbp_next = prev['diastolic_bp'] + p['theta'] * (dbp_mu - prev['diastolic_bp']) + np.random.normal(0, p['sigma'])
        dbp_next = min(dbp_next, sbp_next - 10.0)
        dbp_next = np.clip(dbp_next, p['min'], p['max'])
        
        # Activity step
        p = PROG_PARAMS['activity_mims']
        act_next = prev['activity_mims'] + p['theta'] * (act_mu - prev['activity_mims']) + np.random.normal(0, p['sigma'])
        act_next = np.clip(act_next, p['min'], p['max'])
        
        # Adherence step
        p = PROG_PARAMS['adherence_pct']
        adh_next = prev['adherence_pct'] + p['theta'] * (adh_mu - prev['adherence_pct']) + np.random.normal(0, p['sigma'])
        adh_next = np.clip(adh_next, p['min'], p['max'])
        
        # Pain step
        p = PROG_PARAMS['pain_score']
        pain_next = prev['pain_score'] + p['theta'] * (pain_mu - prev['pain_score']) + np.random.normal(0, p['sigma'])
        pain_next = np.clip(pain_next, p['min'], p['max'])
        
        # Heart rate step
        p = PROG_PARAMS['heart_rate']
        hr_next = prev['heart_rate'] + p['theta'] * (hr_mu - prev['heart_rate']) + np.random.normal(0, p['sigma'])
        hr_next = np.clip(hr_next, p['min'], p['max'])
        
        curr = {
            'month': step if cadence == "monthly" else step // 4,
            'day_offset': step * step_days,
            'age': m0['age'],
            'sex': m0['sex'],
            'diabetes': m0['diabetes'],
            'systolic_bp': round(float(sbp_next), 1),
            'diastolic_bp': round(float(dbp_next), 1),
            'activity_mims': round(float(act_next), 1),
            'n_medications': m0['n_medications'],
            'adherence_pct': round(float(adh_next), 1),
            'pain_score': round(float(pain_next), 1),
            'heart_rate': round(float(hr_next), 1)
        }
        trajectory.append(curr)
        
    return {
        'patient_id': patient_dict.get('patient_id', 'SYN-000001'),
        'cadence': cadence,
        'num_months': num_months,
        'trajectory': trajectory,
        'disclaimer': 'This is a synthetic population trajectory, not a clinical prediction for an individual.'
    }
