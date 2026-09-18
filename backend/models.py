import datetime
from sqlalchemy import Column, String, Float, Integer, Text, DateTime
from backend.database import Base

class MIAAttackDB(Base):
    __tablename__ = "mia_attacks"

    attack_id = Column(String, primary_key=True, index=True)
    cohort_id = Column(String, index=True)
    attacker_model = Column(String)
    auc = Column(Float)
    precision_at_50 = Column(Float)
    recall_at_50 = Column(Float)
    roc_curve = Column(Text)  # JSON text
    verdict = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class EdgeCaseScenarioDB(Base):
    __tablename__ = "edge_case_scenarios"

    scenario_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    target_constraints = Column(Text)  # JSON text

class BiasAuditResultDB(Base):
    __tablename__ = "bias_audit_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(String, index=True)
    cohort_id = Column(String, index=True)
    reference = Column(String)
    subgroup = Column(String)
    real_pct = Column(Float)
    synthetic_pct = Column(Float)
    gap = Column(Float)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class APIKeyDB(Base):
    __tablename__ = "api_keys"

    key_id = Column(String, primary_key=True, index=True)
    api_key_hash = Column(String, unique=True, index=True)
    label = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_used_at = Column(DateTime, default=datetime.datetime.utcnow)
    request_count = Column(Integer, default=0)

class GeneratedCohortDB(Base):
    __tablename__ = "generated_cohorts"

    cohort_id = Column(String, primary_key=True, index=True)
    n_generated = Column(Integer)
    provenance = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
