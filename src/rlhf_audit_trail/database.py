"""Database models and connection management for audit trail storage."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

try:
    from sqlalchemy import (
        Column, Integer, String, Float, DateTime, JSON, Boolean, Text,
        ForeignKey, Index, create_engine, func
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.pool import StaticPool
    import sqlalchemy as sa
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .exceptions import AuditTrailError

logger = logging.getLogger(__name__)

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class AuditEvent(Base):
        """Database model for audit events."""
        __tablename__ = 'audit_events'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(36), nullable=False, index=True)
        event_type = Column(String(50), nullable=False, index=True)
        timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
        event_data = Column(JSON, nullable=False)
        signature = Column(Text, nullable=True)
        merkle_hash = Column(String(64), nullable=True)
        verified = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Indexes for common queries
        __table_args__ = (
            Index('idx_session_timestamp', 'session_id', 'timestamp'),
            Index('idx_event_type_timestamp', 'event_type', 'timestamp'),
        )
        
        def __repr__(self):
            return f"<AuditEvent(id={self.id}, session_id='{self.session_id}', event_type='{self.event_type}')>"

    class TrainingSession(Base):
        """Database model for training sessions."""
        __tablename__ = 'training_sessions'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(36), unique=True, nullable=False, index=True)
        experiment_name = Column(String(255), nullable=False)
        model_name = Column(String(255), nullable=False)
        start_time = Column(DateTime, nullable=False, index=True)
        end_time = Column(DateTime, nullable=True)
        status = Column(String(50), nullable=False, default='active')
        metadata = Column(JSON, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Relationship to audit events
        events = relationship("AuditEvent", backref="session", lazy='dynamic',
                            primaryjoin="TrainingSession.session_id == foreign(AuditEvent.session_id)")
        
        def __repr__(self):
            return f"<TrainingSession(session_id='{self.session_id}', experiment='{self.experiment_name}')>"

    class AnnotationBatch(Base):
        """Database model for annotation batches."""
        __tablename__ = 'annotation_batches'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        batch_id = Column(String(36), unique=True, nullable=False, index=True)
        session_id = Column(String(36), nullable=False, index=True)
        batch_size = Column(Integer, nullable=False)
        privacy_cost = Column(Float, nullable=False)
        timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
        metadata = Column(JSON, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        def __repr__(self):
            return f"<AnnotationBatch(batch_id='{self.batch_id}', size={self.batch_size})>"

    class PolicyUpdate(Base):
        """Database model for policy updates."""
        __tablename__ = 'policy_updates'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        update_id = Column(String(36), unique=True, nullable=False, index=True)
        session_id = Column(String(36), nullable=False, index=True)
        step_number = Column(Integer, nullable=False)
        loss = Column(Float, nullable=False)
        parameter_delta_norm = Column(Float, nullable=False)
        gradient_norm = Column(Float, nullable=False)
        learning_rate = Column(Float, nullable=False)
        timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
        metadata = Column(JSON, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        def __repr__(self):
            return f"<PolicyUpdate(update_id='{self.update_id}', step={self.step_number})>"

    class ComplianceReport(Base):
        """Database model for compliance reports."""
        __tablename__ = 'compliance_reports'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        report_id = Column(String(36), unique=True, nullable=False, index=True)
        session_id = Column(String(36), nullable=False, index=True)
        framework = Column(String(50), nullable=False)
        compliance_score = Column(Float, nullable=False)
        is_compliant = Column(Boolean, nullable=False)
        issues = Column(JSON, nullable=True)
        recommendations = Column(JSON, nullable=True)
        timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
        metadata = Column(JSON, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        def __repr__(self):
            return f"<ComplianceReport(report_id='{self.report_id}', framework='{self.framework}')>"

    class PrivacyBudgetUsage(Base):
        """Database model for privacy budget tracking."""
        __tablename__ = 'privacy_budget_usage'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(36), nullable=False, index=True)
        operation_type = Column(String(50), nullable=False)
        epsilon_cost = Column(Float, nullable=False)
        delta_cost = Column(Float, nullable=False)
        total_epsilon_used = Column(Float, nullable=False)
        total_delta_used = Column(Float, nullable=False)
        remaining_epsilon = Column(Float, nullable=False)
        remaining_delta = Column(Float, nullable=False)
        timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
        metadata = Column(JSON, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        def __repr__(self):
            return f"<PrivacyBudgetUsage(session_id='{self.session_id}', epsilon_cost={self.epsilon_cost})>"


class DatabaseManager:
    """Manages database connections and operations for audit trail storage."""
    
    def __init__(self, database_url: str = None, echo: bool = False, pool_size: int = 10):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL (defaults to SQLite)
            echo: Whether to echo SQL statements for debugging
            pool_size: Connection pool size
        """
        if not SQLALCHEMY_AVAILABLE:
            raise AuditTrailError("SQLAlchemy not available. Install with: pip install sqlalchemy")
        
        if database_url is None:
            # Default to SQLite for development
            database_url = "sqlite:///./audit_trail.db"
            
        self.database_url = database_url
        self.echo = echo
        
        # Create engine with appropriate settings
        if database_url.startswith("sqlite"):
            # SQLite-specific settings
            self.engine = create_engine(
                database_url,
                echo=echo,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False}
            )
        else:
            # PostgreSQL/MySQL settings
            self.engine = create_engine(
                database_url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=300
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize database
        self.init_database()
        
        logger.info(f"Initialized database manager with URL: {self._mask_database_url(database_url)}")
    
    def _mask_database_url(self, url: str) -> str:
        """Mask sensitive information in database URL for logging."""
        if "://" in url:
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                credentials, host_part = rest.split("@", 1)
                return f"{protocol}://***:***@{host_part}"
        return url
    
    def init_database(self):
        """Initialize database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise AuditTrailError(f"Database initialization failed: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session."""
        return self.SessionLocal()
    
    async def store_audit_event(self, event_data: Dict[str, Any]) -> int:
        """Store an audit event in the database.
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            Event ID
        """
        async with self.get_session() as session:
            event = AuditEvent(
                session_id=event_data['session_id'],
                event_type=event_data['event_type'],
                timestamp=datetime.fromtimestamp(event_data['timestamp']),
                event_data=event_data['data'],
                signature=event_data.get('signature'),
                merkle_hash=event_data.get('merkle_hash'),
                verified=event_data.get('verified', False)
            )
            
            session.add(event)
            session.flush()
            event_id = event.id
            
            logger.debug(f"Stored audit event {event_id} for session {event_data['session_id']}")
            return event_id
    
    async def store_training_session(self, session_data: Dict[str, Any]) -> int:
        """Store a training session record.
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            Session record ID
        """
        async with self.get_session() as session:
            training_session = TrainingSession(
                session_id=session_data['session_id'],
                experiment_name=session_data['experiment_name'],
                model_name=session_data['model_name'],
                start_time=datetime.fromtimestamp(session_data['start_time']),
                end_time=datetime.fromtimestamp(session_data['end_time']) if session_data.get('end_time') else None,
                status=session_data.get('status', 'active'),
                metadata=session_data.get('metadata', {})
            )
            
            session.add(training_session)
            session.flush()
            record_id = training_session.id
            
            logger.debug(f"Stored training session {record_id} with session_id {session_data['session_id']}")
            return record_id
    
    async def get_session_events(self, session_id: str, 
                                event_type: Optional[str] = None,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None,
                                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve audit events for a session.
        
        Args:
            session_id: Training session ID
            event_type: Optional event type filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Optional limit on number of events
            
        Returns:
            List of event dictionaries
        """
        async with self.get_session() as session:
            query = session.query(AuditEvent).filter(AuditEvent.session_id == session_id)
            
            if event_type:
                query = query.filter(AuditEvent.event_type == event_type)
            
            if start_time:
                query = query.filter(AuditEvent.timestamp >= start_time)
                
            if end_time:
                query = query.filter(AuditEvent.timestamp <= end_time)
            
            query = query.order_by(AuditEvent.timestamp.asc())
            
            if limit:
                query = query.limit(limit)
            
            events = query.all()
            
            return [
                {
                    'id': event.id,
                    'session_id': event.session_id,
                    'event_type': event.event_type,
                    'timestamp': event.timestamp,
                    'event_data': event.event_data,
                    'signature': event.signature,
                    'merkle_hash': event.merkle_hash,
                    'verified': event.verified
                }
                for event in events
            ]
    
    async def get_privacy_budget_usage(self, session_id: str) -> List[Dict[str, Any]]:
        """Get privacy budget usage for a session.
        
        Args:
            session_id: Training session ID
            
        Returns:
            List of privacy budget usage records
        """
        async with self.get_session() as session:
            usage_records = session.query(PrivacyBudgetUsage).filter(
                PrivacyBudgetUsage.session_id == session_id
            ).order_by(PrivacyBudgetUsage.timestamp.asc()).all()
            
            return [
                {
                    'operation_type': record.operation_type,
                    'epsilon_cost': record.epsilon_cost,
                    'delta_cost': record.delta_cost,
                    'total_epsilon_used': record.total_epsilon_used,
                    'total_delta_used': record.total_delta_used,
                    'remaining_epsilon': record.remaining_epsilon,
                    'remaining_delta': record.remaining_delta,
                    'timestamp': record.timestamp,
                    'metadata': record.metadata
                }
                for record in usage_records
            ]
    
    async def get_training_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a training session.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Dictionary with session statistics
        """
        async with self.get_session() as session:
            # Get session info
            training_session = session.query(TrainingSession).filter(
                TrainingSession.session_id == session_id
            ).first()
            
            if not training_session:
                return {}
            
            # Get event counts by type
            event_counts = session.query(
                AuditEvent.event_type,
                func.count(AuditEvent.id).label('count')
            ).filter(
                AuditEvent.session_id == session_id
            ).group_by(AuditEvent.event_type).all()
            
            # Get annotation batch info
            annotation_stats = session.query(
                func.count(AnnotationBatch.id).label('batch_count'),
                func.sum(AnnotationBatch.batch_size).label('total_annotations'),
                func.sum(AnnotationBatch.privacy_cost).label('total_privacy_cost')
            ).filter(AnnotationBatch.session_id == session_id).first()
            
            # Get policy update count
            update_count = session.query(PolicyUpdate).filter(
                PolicyUpdate.session_id == session_id
            ).count()
            
            return {
                'session_id': session_id,
                'experiment_name': training_session.experiment_name,
                'model_name': training_session.model_name,
                'start_time': training_session.start_time,
                'end_time': training_session.end_time,
                'status': training_session.status,
                'duration': (training_session.end_time - training_session.start_time).total_seconds() if training_session.end_time else None,
                'event_counts': {event_type: count for event_type, count in event_counts},
                'annotation_batch_count': annotation_stats.batch_count or 0,
                'total_annotations': annotation_stats.total_annotations or 0,
                'total_privacy_cost': annotation_stats.total_privacy_cost or 0.0,
                'policy_update_count': update_count,
                'metadata': training_session.metadata
            }
    
    async def cleanup_old_sessions(self, retention_days: int = 90):
        """Clean up old training sessions and related data.
        
        Args:
            retention_days: Number of days to retain data
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        async with self.get_session() as session:
            # Get session IDs to delete
            old_sessions = session.query(TrainingSession.session_id).filter(
                TrainingSession.created_at < cutoff_date
            ).all()
            
            session_ids = [s.session_id for s in old_sessions]
            
            if not session_ids:
                logger.info("No old sessions to clean up")
                return
            
            # Delete related data
            for table in [AuditEvent, AnnotationBatch, PolicyUpdate, ComplianceReport, PrivacyBudgetUsage]:
                deleted = session.query(table).filter(table.session_id.in_(session_ids)).delete(synchronize_session=False)
                logger.info(f"Deleted {deleted} records from {table.__tablename__}")
            
            # Delete training sessions
            deleted_sessions = session.query(TrainingSession).filter(
                TrainingSession.session_id.in_(session_ids)
            ).delete(synchronize_session=False)
            
            logger.info(f"Cleaned up {deleted_sessions} old training sessions")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check.
        
        Returns:
            Health check results
        """
        try:
            async with self.get_session() as session:
                # Test basic connectivity
                result = session.execute("SELECT 1").fetchone()
                
                # Get basic statistics
                session_count = session.query(TrainingSession).count()
                event_count = session.query(AuditEvent).count()
                
                return {
                    'status': 'healthy',
                    'database_url': self._mask_database_url(self.database_url),
                    'connection_test': result[0] == 1,
                    'total_sessions': session_count,
                    'total_events': event_count,
                    'timestamp': datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow()
            }

else:
    # Dummy implementations when SQLAlchemy is not available
    class DatabaseManager:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("SQLAlchemy not available. Install with: pip install sqlalchemy")
        
        async def health_check(self):
            return {'status': 'unavailable', 'error': 'SQLAlchemy not installed'}