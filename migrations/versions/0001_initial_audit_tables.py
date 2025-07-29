"""Initial audit trail tables

Revision ID: 0001
Revises: 
Create Date: 2025-07-29 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial audit trail tables."""
    
    # Create audit_sessions table
    op.create_table(
        'audit_sessions',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('experiment_name', sa.String(255), nullable=False),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Enum('active', 'completed', 'failed', name='session_status'), nullable=False),
        sa.Column('config', postgresql.JSONB(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create audit_events table
    op.create_table(
        'audit_events',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('audit_sessions.id'), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('event_data', postgresql.JSONB(), nullable=False),
        sa.Column('integrity_hash', sa.String(64), nullable=False),
        sa.Column('previous_hash', sa.String(64), nullable=True),
        sa.Column('merkle_root', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create annotations table for RLHF data
    op.create_table(
        'annotations',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('audit_sessions.id'), nullable=False),
        sa.Column('prompt_hash', sa.String(64), nullable=False),
        sa.Column('response_hash', sa.String(64), nullable=False),
        sa.Column('annotator_id', sa.String(64), nullable=False),  # Anonymized ID
        sa.Column('reward', sa.Float(), nullable=False),
        sa.Column('privacy_noise', sa.Float(), nullable=True),
        sa.Column('annotation_data', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create model_checkpoints table
    op.create_table(
        'model_checkpoints',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('audit_sessions.id'), nullable=False),
        sa.Column('checkpoint_name', sa.String(255), nullable=False),
        sa.Column('epoch', sa.Integer(), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('metrics', postgresql.JSONB(), nullable=True),
        sa.Column('parameter_delta_norm', sa.Float(), nullable=True),
        sa.Column('storage_path', sa.String(512), nullable=True),
        sa.Column('storage_hash', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create privacy_budgets table
    op.create_table(
        'privacy_budgets',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('audit_sessions.id'), nullable=False),
        sa.Column('annotator_id', sa.String(64), nullable=False),
        sa.Column('epsilon_used', sa.Float(), nullable=False, default=0.0),
        sa.Column('delta_used', sa.Float(), nullable=False, default=0.0),
        sa.Column('total_epsilon_budget', sa.Float(), nullable=False),
        sa.Column('total_delta_budget', sa.Float(), nullable=False),
        sa.Column('operations_count', sa.Integer(), nullable=False, default=0),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create compliance_reports table
    op.create_table(
        'compliance_reports',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('session_id', sa.String(64), sa.ForeignKey('audit_sessions.id'), nullable=True),
        sa.Column('report_type', sa.String(50), nullable=False),  # 'eu_ai_act', 'nist', etc.
        sa.Column('report_data', postgresql.JSONB(), nullable=False),
        sa.Column('compliance_status', sa.Enum('compliant', 'non_compliant', 'pending', name='compliance_status'), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create indexes for performance
    op.create_index('idx_audit_events_session_timestamp', 'audit_events', ['session_id', 'timestamp'])
    op.create_index('idx_audit_events_type', 'audit_events', ['event_type'])
    op.create_index('idx_annotations_session', 'annotations', ['session_id'])
    op.create_index('idx_annotations_annotator', 'annotations', ['annotator_id'])
    op.create_index('idx_checkpoints_session_step', 'model_checkpoints', ['session_id', 'step'])
    op.create_index('idx_privacy_budgets_annotator', 'privacy_budgets', ['annotator_id'])
    op.create_index('idx_compliance_reports_type_status', 'compliance_reports', ['report_type', 'compliance_status'])


def downgrade() -> None:
    """Drop all audit trail tables."""
    
    # Drop indexes first
    op.drop_index('idx_compliance_reports_type_status')
    op.drop_index('idx_privacy_budgets_annotator')
    op.drop_index('idx_checkpoints_session_step')
    op.drop_index('idx_annotations_annotator')
    op.drop_index('idx_annotations_session')
    op.drop_index('idx_audit_events_type')
    op.drop_index('idx_audit_events_session_timestamp')
    
    # Drop tables in reverse dependency order
    op.drop_table('compliance_reports')
    op.drop_table('privacy_budgets')
    op.drop_table('model_checkpoints')
    op.drop_table('annotations')
    op.drop_table('audit_events')
    op.drop_table('audit_sessions')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS compliance_status')
    op.execute('DROP TYPE IF EXISTS session_status')