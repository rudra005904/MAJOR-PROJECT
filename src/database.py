"""Database connection and examination history tracking.

Supports both:
- Cloud PostgreSQL (Supabase, Neon, AWS RDS) via DATABASE_URL
- Local SQLite fallback (data/biometrics.db) for zero-configuration offline use
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# Read DATABASE_URL or default to local SQLite database in data/
DEFAULT_SQLITE_PATH = Path("data") / "biometrics.db"
DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.resolve()}"

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

# Supabase pooler compatibility: convert postgres:// to postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite connect_args
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception:
    # Fail-safe fallback to SQLite if connection string has invalid credentials
    engine = create_engine(DEFAULT_SQLITE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class BiometricExamination(Base):
    """Stores all forensic fingerprint classification examinations."""

    __tablename__ = "biometric_examinations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    examination_id = Column(String(64), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(32), default="BMP")
    filesize_formatted = Column(String(64), default="64.0 KB")
    predicted_class = Column(String(32), index=True, nullable=False)
    predicted_class_title = Column(String(64), nullable=False)
    confidence_score = Column(Float, nullable=False)
    confidence_percentage = Column(String(32), nullable=False)
    probabilities_json = Column(Text, nullable=False)
    is_borderline = Column(Boolean, default=False)
    latency_ms = Column(Float, default=0.0)
    pdf_info = Column(String(255), nullable=True)
    source = Column(String(32), default="upload")  # "upload" or "preset"
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Converts model instance to JSON-serializable dictionary."""
        probs = {}
        if self.probabilities_json:
            try:
                probs = json.loads(self.probabilities_json)
            except Exception:
                probs = {}

        return {
            "id": self.id,
            "examination_id": self.examination_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "filesize_formatted": self.filesize_formatted,
            "predicted_class": self.predicted_class,
            "predicted_class_title": self.predicted_class_title,
            "confidence_score": self.confidence_score,
            "confidence_percentage": self.confidence_percentage,
            "probabilities": probs,
            "is_borderline": self.is_borderline,
            "latency_ms": self.latency_ms,
            "pdf_info": self.pdf_info,
            "source": self.source,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else "",
        }


def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)


def log_examination(data: Dict[str, Any]) -> BiometricExamination:
    """Logs a new classification result to the database."""
    init_db()
    session = SessionLocal()
    try:
        exam_id = "PU-BIO-" + str(int(time.time() * 1000))[-8:]
        probs = data.get("probabilities", {})
        conf_score = probs.get(data.get("predicted_class", ""), 0.0)
        is_borderline = data.get("borderline_analysis", {}).get("is_borderline", False)

        record = BiometricExamination(
            examination_id=exam_id,
            filename=data.get("filename", "unknown_scan"),
            file_type=data.get("file_type", "BMP"),
            filesize_formatted=data.get("filesize_formatted", "N/A"),
            predicted_class=data.get("predicted_class", "unknown"),
            predicted_class_title=data.get("predicted_class_title", "Unknown Pattern"),
            confidence_score=round(conf_score, 4),
            confidence_percentage=data.get("confidence_percentage", f"{conf_score*100:.2f}%"),
            probabilities_json=json.dumps(probs),
            is_borderline=is_borderline,
            latency_ms=data.get("latency_ms", 0.0),
            pdf_info=data.get("pdf_info"),
            source=data.get("source", "upload"),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    except Exception as exc:
        session.rollback()
        raise exc
    finally:
        session.close()


def list_examinations(
    limit: int = 50,
    offset: int = 0,
    pattern_filter: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves paginated examination records with optional filtering."""
    init_db()
    session = SessionLocal()
    try:
        query = session.query(BiometricExamination)

        if pattern_filter and pattern_filter.lower() in ("arch", "loop", "whorl"):
            query = query.filter(BiometricExamination.predicted_class == pattern_filter.lower())
        elif pattern_filter and pattern_filter.lower() == "borderline":
            query = query.filter(BiometricExamination.is_borderline == True)

        if search:
            query = query.filter(BiometricExamination.filename.ilike(f"%{search}%"))

        total_count = query.count()
        records = query.order_by(desc(BiometricExamination.created_at)).offset(offset).limit(limit).all()

        return {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "records": [r.to_dict() for r in records],
        }
    finally:
        session.close()


def get_examination_by_id(exam_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single examination by its unique examination ID."""
    init_db()
    session = SessionLocal()
    try:
        record = session.query(BiometricExamination).filter(BiometricExamination.examination_id == exam_id).first()
        return record.to_dict() if record else None
    finally:
        session.close()


def get_db_stats() -> Dict[str, Any]:
    """Returns database summary statistics."""
    init_db()
    session = SessionLocal()
    try:
        total = session.query(BiometricExamination).count()
        arches = session.query(BiometricExamination).filter(BiometricExamination.predicted_class == "arch").count()
        loops = session.query(BiometricExamination).filter(BiometricExamination.predicted_class == "loop").count()
        whorls = session.query(BiometricExamination).filter(BiometricExamination.predicted_class == "whorl").count()
        borderlines = session.query(BiometricExamination).filter(BiometricExamination.is_borderline == True).count()
        return {
            "total_examinations": total,
            "arch_count": arches,
            "loop_count": loops,
            "whorl_count": whorls,
            "borderline_count": borderlines,
            "database_type": "PostgreSQL (Supabase/Cloud)" if "postgresql" in DATABASE_URL else "SQLite (Local/Portable)",
        }
    finally:
        session.close()
