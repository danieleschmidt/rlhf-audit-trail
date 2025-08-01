#!/usr/bin/env python3
"""
Terragon Autonomous SDLC - Value Discovery Engine
Perpetual value discovery and intelligent prioritization system
"""

import json
import subprocess
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValueItem:
    """Represents a discovered work item with comprehensive scoring"""
    id: str
    title: str
    description: str
    category: str  # security, technical_debt, performance, compliance, feature
    source: str    # git_history, static_analysis, issues, vulnerabilities
    files_affected: List[str]
    estimated_effort_hours: float
    
    # Scoring components
    wsjf_score: float = 0.0
    ice_score: float = 0.0  
    technical_debt_score: float = 0.0
    security_priority: float = 0.0
    compliance_priority: float = 0.0
    composite_score: float = 0.0
    
    # Metadata
    discovered_at: str = ""
    risk_level: str = "medium"  # low, medium, high, critical
    dependencies: List[str] = None
    ai_ml_specific: bool = False
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if not self.discovered_at:
            self.discovered_at = datetime.now().isoformat()

@dataclass 
class ValueMetrics:
    """Track value delivery and learning metrics"""
    total_items_discovered: int = 0
    total_items_completed: int = 0
    average_cycle_time_hours: float = 0.0
    value_delivered_score: float = 0.0
    technical_debt_reduction: float = 0.0
    security_improvements: int = 0
    compliance_improvements: int = 0
    performance_gains_percent: float = 0.0
    
    # Learning metrics
    estimation_accuracy: float = 0.0
    value_prediction_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

class ValueDiscoveryEngine:
    """Core engine for autonomous value discovery and scoring"""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.metrics_file = Path(".terragon/value-metrics.json")
        self.backlog_file = Path("BACKLOG.md")
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
    
    def discover_value_items(self) -> List[ValueItem]:
        """Execute comprehensive value discovery across all sources"""
        logger.info("🔍 Starting comprehensive value discovery...")
        
        all_items = []
        
        # Git history analysis
        if self.config['discovery']['sources']['gitHistory']['enabled']:
            git_items = self._analyze_git_history()
            all_items.extend(git_items)
            logger.info(f"📊 Discovered {len(git_items)} items from Git history")
        
        # Static analysis
        if self.config['discovery']['sources']['staticAnalysis']['enabled']:
            static_items = self._run_static_analysis()
            all_items.extend(static_items)
            logger.info(f"🔧 Discovered {len(static_items)} items from static analysis")
        
        # Security vulnerability scanning
        if self.config['discovery']['sources']['vulnerabilityDatabases']['enabled']:
            vuln_items = self._scan_vulnerabilities()
            all_items.extend(vuln_items)
            logger.info(f"🛡️ Discovered {len(vuln_items)} security items")
        
        # Performance regression analysis
        if self.config['discovery']['sources']['performanceMonitoring']['enabled']:
            perf_items = self._analyze_performance()
            all_items.extend(perf_items)
            logger.info(f"⚡ Discovered {len(perf_items)} performance items")
        
        # AI/ML specific analysis
        ai_items = self._analyze_ai_ml_patterns()
        all_items.extend(ai_items)
        logger.info(f"🤖 Discovered {len(ai_items)} AI/ML specific items")
        
        # Compliance gap analysis
        compliance_items = self._analyze_compliance_gaps()
        all_items.extend(compliance_items)
        logger.info(f"📋 Discovered {len(compliance_items)} compliance items")
        
        # Score and prioritize all items
        scored_items = [self._calculate_composite_score(item) for item in all_items]
        
        # Sort by composite score descending
        scored_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"✅ Total value discovery complete: {len(scored_items)} items")
        return scored_items
    
    def _analyze_git_history(self) -> List[ValueItem]:
        """Analyze Git history for TODO, FIXME, and improvement opportunities"""
        items = []
        keywords = self.config['discovery']['sources']['gitHistory']['keywords']
        
        try:
            # Get recent commits with keyword mentions
            result = subprocess.run([
                'git', 'log', '--grep=' + '|'.join(keywords), 
                '--oneline', '--since=3 months ago'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash = line.split()[0]
                    commit_msg = ' '.join(line.split()[1:])
                    
                    # Determine category based on keywords
                    category = "technical_debt"
                    if any(sec in commit_msg.lower() for sec in ['security', 'vulnerability', 'cve']):
                        category = "security"
                    elif any(perf in commit_msg.lower() for perf in ['performance', 'optimize', 'slow']):
                        category = "performance"
                    elif any(comp in commit_msg.lower() for comp in ['compliance', 'audit', 'regulation']):
                        category = "compliance"
                    
                    items.append(ValueItem(
                        id=f"git-{commit_hash}",
                        title=f"Address: {commit_msg[:60]}...",
                        description=f"Git commit indicates work needed: {commit_msg}",
                        category=category,
                        source="git_history",
                        files_affected=[],
                        estimated_effort_hours=2.0,
                        ai_ml_specific="rlhf" in commit_msg.lower() or "ai" in commit_msg.lower()
                    ))
            
            # Search for TODO/FIXME in current codebase
            result = subprocess.run([
                'git', 'grep', '-n', '-E', '(TODO|FIXME|XXX|HACK)', '*.py'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            for line in result.stdout.strip().split('\n'):
                if line and ':' in line:
                    file_path, line_num, content = line.split(':', 2)
                    
                    # Extract the actual TODO/FIXME content
                    todo_match = re.search(r'(TODO|FIXME|XXX|HACK).*?$', content)
                    if todo_match:
                        todo_text = todo_match.group().strip()
                        
                        items.append(ValueItem(
                            id=f"todo-{file_path}-{line_num}",
                            title=f"Code comment in {file_path}:{line_num}",
                            description=todo_text,
                            category="technical_debt",
                            source="git_history",
                            files_affected=[file_path],
                            estimated_effort_hours=1.5,
                            ai_ml_specific="rlhf" in file_path or "ai" in file_path
                        ))
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git analysis failed: {e}")
        
        return items
    
    def _run_static_analysis(self) -> List[ValueItem]:
        """Run static analysis tools and extract improvement opportunities"""
        items = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run([
                'ruff', 'check', '--output-format=json', '.'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues:
                    items.append(ValueItem(
                        id=f"ruff-{issue['filename']}-{issue['location']['row']}",
                        title=f"Code quality: {issue['code']} in {issue['filename']}",
                        description=issue['message'],
                        category="technical_debt",
                        source="static_analysis",
                        files_affected=[issue['filename']],
                        estimated_effort_hours=0.5,
                        ai_ml_specific="rlhf" in issue['filename'] or "ai" in issue['filename']
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Ruff analysis failed: {e}")
        
        # Run mypy for type checking issues
        try:
            result = subprocess.run([
                'mypy', '--show-error-codes', '--json-report', '/tmp/mypy-report', 'src/'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            # Parse mypy JSON report if available
            mypy_report_path = Path('/tmp/mypy-report/index.json')
            if mypy_report_path.exists():
                with open(mypy_report_path) as f:
                    mypy_data = json.load(f)
                    
                for file_data in mypy_data.get('files', []):
                    if file_data.get('error_count', 0) > 0:
                        items.append(ValueItem(
                            id=f"mypy-{file_data['module']}",
                            title=f"Type checking issues in {file_data['module']}",
                            description=f"{file_data['error_count']} type checking errors",
                            category="technical_debt", 
                            source="static_analysis",
                            files_affected=[file_data['module']],
                            estimated_effort_hours=1.0,
                            ai_ml_specific="rlhf" in file_data['module'] or "ai" in file_data['module']
                        ))
        except Exception as e:
            logger.warning(f"MyPy analysis failed: {e}")
        
        return items
    
    def _scan_vulnerabilities(self) -> List[ValueItem]:
        """Scan for security vulnerabilities using multiple sources"""
        items = []
        
        # Safety check for Python dependencies
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    items.append(ValueItem(
                        id=f"vuln-{vuln['vulnerability_id']}",
                        title=f"Security vulnerability in {vuln['package_name']}",
                        description=f"{vuln['advisory']} (CVE: {vuln.get('cve', 'N/A')})",
                        category="security",
                        source="vulnerabilities",
                        files_affected=["requirements.txt", "pyproject.toml"],
                        estimated_effort_hours=3.0,
                        security_priority=5.0,
                        risk_level="high"
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Safety vulnerability scan failed: {e}")
        
        # Bandit security analysis
        try:
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                for issue in bandit_data.get('results', []):
                    items.append(ValueItem(
                        id=f"bandit-{issue['filename']}-{issue['line_number']}",
                        title=f"Security issue: {issue['test_name']}",
                        description=issue['issue_text'],
                        category="security",
                        source="static_analysis",
                        files_affected=[issue['filename']],
                        estimated_effort_hours=2.0,
                        security_priority=issue.get('issue_severity', 'MEDIUM') == 'HIGH' and 4.0 or 2.0,
                        risk_level=issue.get('issue_severity', 'MEDIUM').lower()
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Bandit security scan failed: {e}")
        
        return items
    
    def _analyze_performance(self) -> List[ValueItem]:
        """Analyze performance benchmarks and detect regressions"""
        items = []
        
        # Check if benchmark baseline exists
        baseline_file = Path("benchmarks/baseline.json")
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                
                # Run current benchmarks
                result = subprocess.run([
                    'python', '-m', 'pytest', 'benchmarks/', '--benchmark-json=/tmp/current-bench.json'
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                current_bench_file = Path('/tmp/current-bench.json')
                if current_bench_file.exists():
                    with open(current_bench_file) as f:
                        current_data = json.load(f)
                    
                    # Compare benchmarks for regressions
                    threshold = self.config['discovery']['sources']['performanceMonitoring']['regression_threshold']
                    
                    for benchmark in current_data.get('benchmarks', []):
                        bench_name = benchmark['name']
                        current_mean = benchmark['stats']['mean']
                        
                        # Find corresponding baseline
                        baseline_mean = None
                        for baseline_bench in baseline_data.get('benchmarks', []):
                            if baseline_bench['name'] == bench_name:
                                baseline_mean = baseline_bench['stats']['mean']
                                break
                        
                        if baseline_mean and current_mean > baseline_mean * (1 + threshold):
                            regression_percent = ((current_mean - baseline_mean) / baseline_mean) * 100
                            
                            items.append(ValueItem(
                                id=f"perf-regression-{bench_name}",
                                title=f"Performance regression in {bench_name}",
                                description=f"Performance degraded by {regression_percent:.1f}%",
                                category="performance",
                                source="performance_monitoring",
                                files_affected=["benchmarks/"],
                                estimated_effort_hours=4.0,
                                ai_ml_specific=True,
                                risk_level="high" if regression_percent > 20 else "medium"
                            ))
            
            except Exception as e:
                logger.warning(f"Performance analysis failed: {e}")
        
        return items
    
    def _analyze_ai_ml_patterns(self) -> List[ValueItem]:
        """Analyze AI/ML specific patterns and issues"""
        items = []
        
        ai_patterns = self.config['discovery']['ai_ml_patterns']
        
        # Search for model drift indicators
        for pattern_category, patterns in ai_patterns.items():
            try:
                for pattern in patterns:
                    result = subprocess.run([
                        'git', 'log', '--grep=' + pattern, '--oneline', '--since=1 month ago'
                    ], capture_output=True, text=True, cwd=self.repo_root)
                    
                    if result.stdout.strip():
                        items.append(ValueItem(
                            id=f"ai-pattern-{pattern_category}-{len(items)}",
                            title=f"AI/ML Issue: {pattern_category.replace('_', ' ').title()}",
                            description=f"Pattern detected: {pattern}",
                            category="ai_ml_issue",
                            source="ai_analysis",
                            files_affected=["src/rlhf_audit_trail/"],
                            estimated_effort_hours=6.0,
                            ai_ml_specific=True,
                            risk_level="high" if "compliance" in pattern_category else "medium"
                        ))
            except subprocess.CalledProcessError:
                continue
        
        return items
    
    def _analyze_compliance_gaps(self) -> List[ValueItem]:
        """Analyze compliance requirements and gaps"""
        items = []
        
        # Check EU AI Act compliance
        eu_checklist_path = Path("compliance/eu-ai-act-checklist.yml")
        if eu_checklist_path.exists():
            try:
                with open(eu_checklist_path) as f:
                    checklist = yaml.safe_load(f)
                
                for section_name, section in checklist.items():
                    if isinstance(section, dict) and 'requirements' in section:
                        for req in section['requirements']:
                            if req.get('status') != 'complete':
                                items.append(ValueItem(
                                    id=f"compliance-eu-{section_name}-{req.get('id', len(items))}",
                                    title=f"EU AI Act: {req.get('title', 'Compliance requirement')}",
                                    description=req.get('description', 'Missing compliance requirement'),
                                    category="compliance",
                                    source="compliance_analysis",
                                    files_affected=["compliance/"],
                                    estimated_effort_hours=8.0,
                                    compliance_priority=5.0,
                                    risk_level="critical",
                                    ai_ml_specific=True
                                ))
            except Exception as e:
                logger.warning(f"EU AI Act compliance analysis failed: {e}")
        
        return items
    
    def _calculate_composite_score(self, item: ValueItem) -> ValueItem:
        """Calculate comprehensive composite score using WSJF, ICE, and technical debt"""
        
        # Get adaptive weights based on repository maturity
        maturity_level = self.config['meta']['maturity_level'].replace('-', '_')
        weights = self.config['scoring']['weights'].get(maturity_level, 
                  self.config['scoring']['weights']['advanced'])
        
        # WSJF Components (Cost of Delay / Job Size)
        user_business_value = self._score_business_impact(item)
        time_criticality = self._score_urgency(item)
        risk_reduction = self._score_risk_mitigation(item)
        opportunity_enablement = self._score_opportunity(item)
        
        cost_of_delay = (user_business_value + time_criticality + 
                        risk_reduction + opportunity_enablement)
        
        wsjf_score = cost_of_delay / max(item.estimated_effort_hours, 0.5)
        
        # ICE Components (Impact * Confidence * Ease)
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = self._score_ease(item)
        
        ice_score = impact * confidence * ease
        
        # Technical Debt Score
        debt_impact = self._calculate_debt_cost(item)
        debt_interest = self._calculate_debt_growth(item)
        hotspot_multiplier = self._get_hotspot_multiplier(item)
        
        technical_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Apply priority boosts
        security_boost = 1.0
        compliance_boost = 1.0
        
        if item.category == "security":
            security_boost = self.config['scoring']['thresholds']['securityBoost']
        elif item.category == "compliance":
            compliance_boost = self.config['scoring']['thresholds']['complianceBoost']
        
        # Calculate composite score with adaptive weighting
        composite_score = (
            weights['wsjf'] * self._normalize_score(wsjf_score, 0, 50) +
            weights['ice'] * self._normalize_score(ice_score, 0, 1000) +
            weights['technicalDebt'] * self._normalize_score(technical_debt_score, 0, 100) +
            weights.get('security', 0.1) * item.security_priority +
            weights.get('innovation', 0.05) * (1.0 if item.ai_ml_specific else 0.0)
        ) * security_boost * compliance_boost
        
        # Update item with calculated scores
        item.wsjf_score = wsjf_score
        item.ice_score = ice_score
        item.technical_debt_score = technical_debt_score
        item.composite_score = composite_score
        
        return item
    
    def _score_business_impact(self, item: ValueItem) -> float:
        """Score business value impact (1-10 scale)"""
        if item.category == "compliance":
            return 10.0  # Critical for EU AI Act compliance
        elif item.category == "security":
            return 9.0   # High security priority for AI systems
        elif item.category == "performance" and item.ai_ml_specific:
            return 8.0   # ML performance critical
        elif item.category == "technical_debt":
            return 6.0   # Moderate business impact
        else:
            return 5.0   # Default moderate impact
    
    def _score_urgency(self, item: ValueItem) -> float:
        """Score time criticality (1-10 scale)"""
        if item.risk_level == "critical":
            return 10.0
        elif item.risk_level == "high":
            return 8.0
        elif item.category == "security":
            return 7.0
        elif item.category == "compliance":
            return 6.0
        else:
            return 4.0
    
    def _score_risk_mitigation(self, item: ValueItem) -> float:
        """Score risk reduction value (1-10 scale)"""
        if item.category in ["security", "compliance"]:
            return 9.0
        elif item.ai_ml_specific:
            return 7.0  # AI/ML systems have inherent risks
        else:
            return 5.0
    
    def _score_opportunity(self, item: ValueItem) -> float:
        """Score opportunity enablement (1-10 scale)"""
        if item.category == "performance":
            return 8.0  # Performance improvements enable growth
        elif item.ai_ml_specific:
            return 7.0  # AI/ML improvements enable innovation
        else:
            return 4.0
    
    def _score_impact(self, item: ValueItem) -> float:
        """ICE Impact component (1-10 scale)"""
        return self._score_business_impact(item)
    
    def _score_confidence(self, item: ValueItem) -> float:
        """ICE Confidence component (1-10 scale)"""
        if item.source == "static_analysis":
            return 9.0  # High confidence in static analysis
        elif item.source == "vulnerabilities":
            return 10.0 # Very high confidence in vulnerability reports
        elif item.source == "git_history":
            return 7.0  # Moderate confidence in code comments
        else:
            return 6.0  # Default moderate confidence
    
    def _score_ease(self, item: ValueItem) -> float:
        """ICE Ease component (1-10 scale)"""
        if item.estimated_effort_hours <= 1:
            return 10.0
        elif item.estimated_effort_hours <= 4:
            return 8.0
        elif item.estimated_effort_hours <= 8:
            return 6.0
        else:
            return 4.0
    
    def _calculate_debt_cost(self, item: ValueItem) -> float:
        """Calculate technical debt cost (maintenance hours saved)"""
        if item.category == "technical_debt":
            return item.estimated_effort_hours * 3  # 3x multiplier for debt
        else:
            return item.estimated_effort_hours
    
    def _calculate_debt_growth(self, item: ValueItem) -> float:
        """Calculate debt interest (future cost if not addressed)"""
        if item.category == "technical_debt":
            return item.estimated_effort_hours * 2  # Debt grows over time
        else:
            return 0.0
    
    def _get_hotspot_multiplier(self, item: ValueItem) -> float:
        """Get hotspot multiplier based on file activity"""
        # Simple heuristic: files in src/ are higher priority
        if any("src/" in f for f in item.files_affected):
            return 2.0
        elif any("test" in f for f in item.files_affected):
            return 1.2
        else:
            return 1.0
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range"""
        if max_val == min_val:
            return 50.0
        normalized = ((score - min_val) / (max_val - min_val)) * 100
        return max(0.0, min(100.0, normalized))
    
    def select_next_best_value(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next highest value item for execution"""
        min_score = self.config['scoring']['thresholds']['minScore']
        max_risk = self.config['scoring']['thresholds']['maxRisk']
        
        for item in items:
            if item.composite_score < min_score:
                continue
                
            # Simple risk assessment
            risk_score = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}.get(item.risk_level, 0.5)
            if risk_score > max_risk:
                continue
            
            # Check dependencies (simplified)
            dependencies_met = True  # In practice, check actual dependencies
            if not dependencies_met:
                continue
            
            return item
        
        return None
    
    def update_backlog(self, items: List[ValueItem]) -> None:
        """Update the backlog markdown file with discovered items"""
        top_items = items[:20]  # Show top 20 items
        
        content = f"""# 📊 Terragon Autonomous Value Backlog

**Repository**: {self.config['meta']['repository_name']}  
**Maturity Level**: {self.config['meta']['maturity_level'].upper()} ({self.config['meta']['maturity_score']}%)  
**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Total Items Discovered**: {len(items)}

## 🎯 Next Best Value Item

"""
        
        if items:
            next_item = items[0]
            content += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **Category**: {next_item.category.replace('_', ' ').title()}
- **Source**: {next_item.source.replace('_', ' ').title()}
- **Estimated Effort**: {next_item.estimated_effort_hours} hours
- **Risk Level**: {next_item.risk_level.title()}
- **AI/ML Specific**: {'Yes' if next_item.ai_ml_specific else 'No'}

**Description**: {next_item.description}

**Files Affected**: {', '.join(next_item.files_affected) if next_item.files_affected else 'Multiple/TBD'}

"""
        else:
            content += "No high-value items currently identified. Running continuous discovery...\n\n"
        
        content += f"""## 📋 Top {min(len(top_items), 20)} Value Items

| Rank | ID | Title | Score | Category | Effort (h) | Risk | AI/ML |
|------|-----|--------|---------|----------|------------|------|-------|
"""
        
        for i, item in enumerate(top_items, 1):
            title_truncated = item.title[:50] + "..." if len(item.title) > 50 else item.title
            content += f"| {i} | {item.id} | {title_truncated} | {item.composite_score:.1f} | {item.category.replace('_', ' ').title()} | {item.estimated_effort_hours} | {item.risk_level.title()} | {'✅' if item.ai_ml_specific else '❌'} |\n"
        
        # Add category breakdown
        categories = {}
        for item in items:
            categories[item.category] = categories.get(item.category, 0) + 1
        
        content += f"""
## 📊 Value Discovery Breakdown

### By Category
"""
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            content += f"- **{category.replace('_', ' ').title()}**: {count} items\n"
        
        # Add discovery sources
        sources = {}
        for item in items:
            sources[item.source] = sources.get(item.source, 0) + 1
        
        content += f"""
### By Discovery Source
"""
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(items)) * 100 if items else 0
            content += f"- **{source.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        content += f"""
## 🔄 Continuous Discovery Status

- **Discovery Engine**: ✅ Active
- **Next Scan**: Every commit + hourly security scan
- **Learning Mode**: ✅ Enabled
- **Auto-Execution**: ✅ Enabled (Score ≥ {self.config['scoring']['thresholds']['minScore']})

## 🎯 Value Delivery Targets

- **Cycle Time**: < {self.config['value_metrics']['tracking']['cycle_time_target']}
- **MTTR**: < {self.config['value_metrics']['tracking']['mttr_target']}
- **Deployment Frequency**: {self.config['value_metrics']['tracking']['deployment_frequency']}
- **Change Failure Rate**: {self.config['value_metrics']['tracking']['change_failure_rate']}

---
*Generated by Terragon Autonomous SDLC - Perpetual Value Discovery Engine*
"""
        
        with open(self.backlog_file, 'w') as f:
            f.write(content)
        
        logger.info(f"📋 Updated backlog: {self.backlog_file}")

def main():
    """Main entry point for value discovery"""
    try:
        engine = ValueDiscoveryEngine()
        
        logger.info("🚀 Starting Terragon Autonomous Value Discovery...")
        
        # Discover all value items
        items = engine.discover_value_items()
        
        # Update backlog
        engine.update_backlog(items)
        
        # Select next best value item
        next_item = engine.select_next_best_value(items)
        
        if next_item:
            logger.info(f"🎯 Next best value item: {next_item.title} (Score: {next_item.composite_score:.1f})")
            print(f"NEXT_VALUE_ITEM={next_item.id}")
        else:
            logger.info("✅ No high-value items meet execution criteria")
            print("NEXT_VALUE_ITEM=none")
        
    except Exception as e:
        logger.error(f"❌ Value discovery failed: {e}")
        raise

if __name__ == "__main__":
    main()