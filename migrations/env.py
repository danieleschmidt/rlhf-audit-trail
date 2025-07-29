"""Alembic environment configuration for RLHF Audit Trail.

This module configures the Alembic migration environment for the RLHF Audit Trail
database. It handles both online and offline migration modes and includes proper
audit trail and compliance table configurations.
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import the target metadata from your models
try:
    from rlhf_audit_trail.database.models import Base
    target_metadata = Base.metadata
except ImportError:
    # Fallback if models aren't implemented yet
    target_metadata = None

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def get_database_url() -> str:
    """Get database URL from environment or config."""
    # Priority: Environment variable > config file
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # Fallback to config file
    return config.get_main_option("sqlalchemy.url")

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render schema for tables
        include_schemas=True,
        # Compare types to detect changes
        compare_type=True,
        # Compare server defaults
        compare_server_default=True,
        # Render item order changes
        render_item=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    configuration = config.get_section(config.config_ini_section)
    
    # Override URL from environment if available
    database_url = get_database_url()
    if database_url:
        configuration["sqlalchemy.url"] = database_url
    
    # Create engine with connection pooling
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # Additional engine options for audit trail requirements
        echo=bool(os.getenv("SQL_ECHO", False)),
        future=True,  # SQLAlchemy 2.0 style
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Enhanced migration options
            include_schemas=True,
            compare_type=True,
            compare_server_default=True,
            render_item=True,
            # Audit trail specific options
            transaction_per_migration=True,
            # Include object names in revision
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()

def include_object(object, name, type_, reflected, compare_to):
    """Determine whether to include an object in migrations.
    
    This function can be used to filter out certain objects from
    being included in migrations (e.g., views, temporary tables).
    """
    # Skip views and temporary tables
    if type_ == "table" and (
        name.startswith("temp_") or 
        name.startswith("view_")
    ):
        return False
    
    # Include all other objects
    return True

def render_item(type_, obj, autogen_context):
    """Custom rendering for specific database objects."""
    # Custom rendering can be added here for specific requirements
    # For example, custom handling of audit trail triggers or functions
    return False  # Use default rendering

# Configure context with custom functions
context.configure(
    include_object=include_object,
    render_item=render_item
)

# Run migrations based on context
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()