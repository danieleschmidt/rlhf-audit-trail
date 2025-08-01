#!/usr/bin/env python3
"""
Terragon Autonomous SDLC - Main Orchestrator
Perpetual value discovery and autonomous execution loop
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TerragonOrchestrator:
    """Main orchestrator for autonomous SDLC operations"""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.terragon_dir = Path(".terragon")
        self.discovery_script = self.terragon_dir / "value-discovery.py"
        self.executor_script = self.terragon_dir / "autonomous-executor.py"
        self.state_file = self.terragon_dir / "orchestrator-state.json"
        
    def run_perpetual_loop(self) -> None:
        """Run the perpetual value discovery and execution loop"""
        logger.info("🚀 Starting Terragon Autonomous SDLC Perpetual Loop")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"🔄 Starting iteration {iteration}")
                
                # Run value discovery
                next_item = self._run_value_discovery()
                
                if next_item and next_item != "none":
                    # Execute the highest value item
                    success = self._execute_value_item(next_item)
                    
                    if success:
                        logger.info(f"✅ Successfully executed {next_item}")
                        # Continue immediately to discover new value
                        continue
                    else:
                        logger.warning(f"❌ Execution failed for {next_item}")
                        # Wait before retrying
                        time.sleep(300)  # 5 minutes
                else:
                    logger.info("ℹ️ No high-value items found, waiting...")
                    # No work available, wait before next discovery
                    time.sleep(1800)  # 30 minutes
                
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"❌ Orchestrator error: {e}")
                time.sleep(600)  # 10 minutes before retry
    
    def run_single_iteration(self) -> bool:
        """Run a single iteration of discovery and execution"""
        logger.info("🎯 Running single Terragon iteration")
        
        try:
            # Run value discovery
            next_item = self._run_value_discovery()
            
            if next_item and next_item != "none":
                # Execute the highest value item
                return self._execute_value_item(next_item)
            else:
                logger.info("ℹ️ No high-value items found")
                return True  # Success, just no work
                
        except Exception as e:
            logger.error(f"❌ Single iteration failed: {e}")
            return False
    
    def _run_value_discovery(self) -> Optional[str]:
        """Run value discovery and get next best item"""
        logger.info("🔍 Running value discovery...")
        
        try:
            # Ensure Python path includes current directory
            env = dict(os.environ) if 'os' in globals() else {}
            env['PYTHONPATH'] = str(self.repo_root) + ':' + env.get('PYTHONPATH', '')
            
            result = subprocess.run([
                sys.executable, str(self.discovery_script)
            ], capture_output=True, text=True, cwd=self.repo_root, env=env)
            
            if result.returncode != 0:
                logger.error(f"Value discovery failed: {result.stderr}")
                return None
            
            # Parse output for next value item
            for line in result.stdout.strip().split('\n'):
                if line.startswith('NEXT_VALUE_ITEM='):
                    next_item = line.split('=', 1)[1]
                    logger.info(f"🎯 Next value item: {next_item}")
                    return next_item
            
            logger.warning("No next value item found in discovery output")
            return None
            
        except Exception as e:
            logger.error(f"Value discovery execution failed: {e}")
            return None
    
    def _execute_value_item(self, item_id: str) -> bool:
        """Execute a specific value item"""
        logger.info(f"⚡ Executing value item: {item_id}")
        
        try:
            # Load item data from backlog for execution context
            item_data = self._load_item_data(item_id)
            
            # Ensure Python path includes current directory
            env = dict(os.environ) if 'os' in globals() else {}
            env['PYTHONPATH'] = str(self.repo_root) + ':' + env.get('PYTHONPATH', '')
            
            result = subprocess.run([
                sys.executable, str(self.executor_script), 
                item_id, json.dumps(item_data) if item_data else '{}'
            ], capture_output=True, text=True, cwd=self.repo_root, env=env)
            
            # Parse execution result
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.startswith('SUCCESS:'):
                    pr_url = line.split(':', 1)[1]
                    logger.info(f"✅ Execution successful - PR: {pr_url}")
                    return True
                elif line.startswith('FAILED:'):
                    error_msg = line.split(':', 1)[1]
                    logger.error(f"❌ Execution failed: {error_msg}")
                    return False
                elif line.startswith('ERROR:'):
                    error_msg = line.split(':', 1)[1]
                    logger.error(f"❌ Execution error: {error_msg}")
                    return False
            
            logger.warning(f"Unclear execution result for {item_id}")
            return False
            
        except Exception as e:
            logger.error(f"Value item execution failed: {e}")
            return False
    
    def _load_item_data(self, item_id: str) -> dict:
        """Load item data from backlog or discovery cache"""
        # In a full implementation, this would parse the BACKLOG.md
        # or load from a structured cache
        return {
            'id': item_id,
            'category': 'technical_debt',
            'estimated_effort_hours': 2.0,
            'title': f'Autonomous task: {item_id}',
            'description': f'Automatically discovered task: {item_id}',
            'files_affected': [],
            'ai_ml_specific': False,
            'risk_level': 'medium',
            'composite_score': 50.0,
            'source': 'autonomous_discovery'
        }
    
    def get_status(self) -> dict:
        """Get current orchestrator status"""
        backlog_file = Path("BACKLOG.md")
        execution_log = self.terragon_dir / "execution-log.json"
        
        status = {
            'orchestrator_active': True,
            'last_discovery': 'N/A',
            'total_items_discovered': 0,
            'total_items_executed': 0,
            'success_rate': 0.0,
            'backlog_exists': backlog_file.exists(),
            'execution_log_exists': execution_log.exists()
        }
        
        # Load execution history if available
        if execution_log.exists():
            try:
                with open(execution_log) as f:
                    history = json.load(f)
                    
                status['total_items_executed'] = len(history)
                successful = sum(1 for entry in history if entry['result']['success'])
                status['success_rate'] = successful / len(history) if history else 0.0
                
                if history:
                    status['last_discovery'] = history[-1]['timestamp']
                    
            except Exception as e:
                logger.warning(f"Failed to load execution history: {e}")
        
        # Count items in backlog
        if backlog_file.exists():
            try:
                with open(backlog_file) as f:
                    content = f.read()
                    # Simple count of markdown table rows (approximate)
                    table_rows = content.count('|') // 7  # Assuming 7 columns
                    status['total_items_discovered'] = max(0, table_rows - 2)  # Subtract header rows
            except Exception as e:
                logger.warning(f"Failed to parse backlog: {e}")
        
        return status

def main():
    """Main entry point for orchestrator"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Terragon Autonomous SDLC Orchestrator')
    parser.add_argument('--mode', choices=['perpetual', 'single', 'status'], 
                       default='single', help='Execution mode')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    orchestrator = TerragonOrchestrator()
    
    try:
        if args.mode == 'perpetual':
            orchestrator.run_perpetual_loop()
        elif args.mode == 'single':
            success = orchestrator.run_single_iteration()
            sys.exit(0 if success else 1)
        elif args.mode == 'status':
            status = orchestrator.get_status()
            print(json.dumps(status, indent=2))
        
    except KeyboardInterrupt:
        logger.info("🛑 Orchestrator shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()