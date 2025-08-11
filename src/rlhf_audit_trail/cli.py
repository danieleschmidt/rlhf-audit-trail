"""Command-line interface for RLHF Audit Trail."""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

try:
    import click
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress
    from rich.panel import Panel
    from rich.json import JSON
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .core import AuditableRLHF
from .config import PrivacyConfig, SecurityConfig, ComplianceConfig
from .monitoring import get_monitor
from .database import DatabaseManager
from .api import run_api_server
from .dashboard import run_dashboard
from .exceptions import AuditTrailError

logger = logging.getLogger(__name__)

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class CLIError(Exception):
    """CLI specific errors."""
    pass


def print_status(message: str, status: str = "info"):
    """Print status message with formatting."""
    if RICH_AVAILABLE:
        if status == "success":
            console.print(f"✅ {message}", style="green")
        elif status == "error":
            console.print(f"❌ {message}", style="red")
        elif status == "warning":
            console.print(f"⚠️  {message}", style="yellow")
        else:
            console.print(f"ℹ️  {message}", style="blue")
    else:
        print(f"[{status.upper()}] {message}")


def print_table(data: List[Dict[str, Any]], title: str = "Results"):
    """Print data in table format."""
    if not data:
        print_status("No data to display", "warning")
        return
    
    if RICH_AVAILABLE:
        table = Table(title=title)
        
        # Add columns based on first row
        for key in data[0].keys():
            table.add_column(str(key).title(), justify="left")
        
        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row.values()])
        
        console.print(table)
    else:
        # Fallback to simple printing
        print(f"\n{title}:")
        print("-" * len(title))
        for i, row in enumerate(data):
            print(f"Row {i+1}:")
            for key, value in row.items():
                print(f"  {key}: {value}")
            print()


def print_json(data: Dict[str, Any], title: str = "JSON Data"):
    """Print JSON data with formatting."""
    if RICH_AVAILABLE:
        console.print(Panel(JSON.from_data(data), title=title))
    else:
        print(f"\n{title}:")
        print(json.dumps(data, indent=2, default=str))


async def create_session_command(args):
    """Create a new training session."""
    try:
        privacy_config = PrivacyConfig(
            epsilon=args.epsilon,
            delta=args.delta,
            clip_norm=args.clip_norm,
            noise_multiplier=args.noise_multiplier
        )
        
        security_config = SecurityConfig()
        compliance_config = ComplianceConfig()
        
        auditor = AuditableRLHF(
            model_name=args.model_name,
            privacy_config=privacy_config,
            security_config=security_config,
            compliance_config=compliance_config,
            storage_backend=args.storage_backend,
            compliance_mode=args.compliance_mode
        )
        
        async with auditor.track_training(args.experiment_name) as session:
            print_status(f"Created session: {session.session_id}", "success")
            print_json({
                "session_id": session.session_id,
                "experiment_name": session.experiment_name,
                "model_name": session.model_name,
                "start_time": session.start_time,
                "privacy_config": {
                    "epsilon": privacy_config.epsilon,
                    "delta": privacy_config.delta,
                    "clip_norm": privacy_config.clip_norm
                },
                "compliance_mode": args.compliance_mode
            }, "Session Details")
            
            # Keep session alive for specified duration
            if args.duration > 0:
                print_status(f"Keeping session alive for {args.duration} seconds...")
                await asyncio.sleep(args.duration)
                
    except Exception as e:
        print_status(f"Failed to create session: {e}", "error")
        raise CLIError(str(e))


def status_command(args):
    """Show system status."""
    try:
        monitor = get_monitor()
        status = monitor.get_comprehensive_status()
        
        print_status("System Status Report", "info")
        
        # System metrics
        system = status.get('system', {})
        if system:
            print_json(system, "System Metrics")
        
        # Performance metrics
        performance = status.get('performance', {})
        if performance:
            print_json(performance, "Performance Metrics")
        
        # Audit metrics
        audit = status.get('audit', {})
        if audit:
            print_json(audit, "Audit Metrics")
        
        # Active alerts
        alerts = status.get('active_alerts', [])
        if alerts:
            print_status(f"Active Alerts: {len(alerts)}", "warning")
            for alert in alerts:
                print_json(alert, f"Alert: {alert['name']}")
        else:
            print_status("No active alerts", "success")
            
    except Exception as e:
        print_status(f"Failed to get status: {e}", "error")


def database_command(args):
    """Database management commands."""
    try:
        db_manager = DatabaseManager(
            database_url=args.database_url,
            echo=args.verbose
        )
        
        if args.action == "init":
            db_manager.init_database()
            print_status("Database initialized successfully", "success")
            
        elif args.action == "health":
            health = asyncio.run(db_manager.health_check())
            print_json(health, "Database Health Check")
            
        elif args.action == "cleanup":
            deleted = asyncio.run(db_manager.cleanup_old_sessions(args.retention_days))
            print_status(f"Cleaned up old sessions", "success")
            
    except Exception as e:
        print_status(f"Database operation failed: {e}", "error")


def monitor_command(args):
    """Start monitoring."""
    try:
        monitor = get_monitor()
        
        if args.action == "start":
            monitor.start()
            print_status("Monitoring started", "success")
            
            # Keep running if requested
            if args.daemon:
                print_status("Running in daemon mode. Press Ctrl+C to stop.")
                try:
                    while True:
                        asyncio.sleep(10)
                except KeyboardInterrupt:
                    print_status("Stopping monitoring...", "info")
                    monitor.stop()
                    
        elif args.action == "stop":
            monitor.stop()
            print_status("Monitoring stopped", "success")
            
        elif args.action == "status":
            status = monitor.get_comprehensive_status()
            print_json(status, "Monitoring Status")
            
    except Exception as e:
        print_status(f"Monitoring operation failed: {e}", "error")


def api_command(args):
    """Start API server."""
    try:
        print_status(f"Starting API server on {args.host}:{args.port}", "info")
        run_api_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers
        )
    except Exception as e:
        print_status(f"API server failed: {e}", "error")


def dashboard_command(args):
    """Start dashboard."""
    try:
        print_status(f"Starting dashboard on {args.host}:{args.port}", "info")
        run_dashboard(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except Exception as e:
        print_status(f"Dashboard failed: {e}", "error")


def verify_command(args):
    """Verify audit trail integrity."""
    try:
        # This would integrate with the verification system
        print_status("Audit trail verification not yet implemented", "warning")
        # TODO: Implement verification logic
        
    except Exception as e:
        print_status(f"Verification failed: {e}", "error")


def export_command(args):
    """Export audit data."""
    try:
        print_status("Data export not yet implemented", "warning")
        # TODO: Implement export logic
        
    except Exception as e:
        print_status(f"Export failed: {e}", "error")


def create_parser() -> argparse.ArgumentParser:
    """Create command line parser."""
    parser = argparse.ArgumentParser(
        description="RLHF Audit Trail - Verifiable provenance for RLHF with EU AI Act compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a training session
  rlhf-audit session create --model-name "llama-7b" --experiment-name "safety-v1"
  
  # Start API server
  rlhf-audit api --port 8000
  
  # Start dashboard
  rlhf-audit dashboard --port 8501
  
  # Check system status
  rlhf-audit status
  
  # Initialize database
  rlhf-audit database init
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", help="Log file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Session commands
    session_parser = subparsers.add_parser("session", help="Training session management")
    session_subparsers = session_parser.add_subparsers(dest="session_action")
    
    create_session_parser = session_subparsers.add_parser("create", help="Create training session")
    create_session_parser.add_argument("--model-name", required=True, help="Model name")
    create_session_parser.add_argument("--experiment-name", required=True, help="Experiment name")
    create_session_parser.add_argument("--epsilon", type=float, default=10.0, help="Privacy epsilon")
    create_session_parser.add_argument("--delta", type=float, default=1e-5, help="Privacy delta")
    create_session_parser.add_argument("--clip-norm", type=float, default=1.0, help="Clipping norm")
    create_session_parser.add_argument("--noise-multiplier", type=float, default=1.1, help="Noise multiplier")
    create_session_parser.add_argument("--storage-backend", default="local", choices=["local", "s3", "gcs"])
    create_session_parser.add_argument("--compliance-mode", default="eu_ai_act", choices=["eu_ai_act", "nist_draft", "both"])
    create_session_parser.add_argument("--duration", type=int, default=0, help="Session duration in seconds")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    # Database commands
    db_parser = subparsers.add_parser("database", help="Database management")
    db_parser.add_argument("action", choices=["init", "health", "cleanup"])
    db_parser.add_argument("--database-url", help="Database connection URL")
    db_parser.add_argument("--retention-days", type=int, default=90, help="Data retention days")
    
    # Monitoring commands
    monitor_parser = subparsers.add_parser("monitor", help="System monitoring")
    monitor_parser.add_argument("action", choices=["start", "stop", "status"])
    monitor_parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    # API server
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    api_parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    api_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    # Dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Start dashboard")
    dashboard_parser.add_argument("--host", default="localhost", help="Host to bind")
    dashboard_parser.add_argument("--port", type=int, default=8501, help="Port to bind")
    dashboard_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # Verification
    verify_parser = subparsers.add_parser("verify", help="Verify audit trail integrity")
    verify_parser.add_argument("--session-id", help="Specific session to verify")
    
    # Export
    export_parser = subparsers.add_parser("export", help="Export audit data")
    export_parser.add_argument("--format", choices=["json", "csv", "parquet"], default="json")
    export_parser.add_argument("--output", help="Output file path")
    export_parser.add_argument("--session-id", help="Specific session to export")
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Handle commands
    try:
        if args.command == "session":
            if args.session_action == "create":
                await create_session_command(args)
        elif args.command == "status":
            status_command(args)
        elif args.command == "database":
            database_command(args)
        elif args.command == "monitor":
            monitor_command(args)
        elif args.command == "api":
            api_command(args)
        elif args.command == "dashboard":
            dashboard_command(args)
        elif args.command == "verify":
            verify_command(args)
        elif args.command == "export":
            export_command(args)
        else:
            parser.print_help()
            return 1
            
        return 0
        
    except CLIError as e:
        print_status(f"Command failed: {e}", "error")
        return 1
    except KeyboardInterrupt:
        print_status("Operation cancelled", "warning")
        return 130
    except Exception as e:
        logger.exception("Unexpected error")
        print_status(f"Unexpected error: {e}", "error")
        return 1


def cli_main():
    """Synchronous entry point for setuptools."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        return 130
    except Exception as e:
        print(f"CLI Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())