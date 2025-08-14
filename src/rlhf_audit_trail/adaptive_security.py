"""Adaptive Security System for RLHF Audit Trail.

Implements intelligent threat detection, adaptive security controls,
and dynamic risk assessment with ML-powered anomaly detection.
"""

import asyncio
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
import logging
from contextlib import asynccontextmanager

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy
    class MockNumpy:
        def mean(self, data): return sum(data) / len(data) if data else 0
        def std(self, data): 
            if not data: return 0
            mean_val = self.mean(data)
            return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
        def percentile(self, data, p):
            if not data: return 0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            return sorted_data[int(k)]
        def random(self):
            import random
            class MockRandom:
                def uniform(self, low, high): return random.uniform(low, high)
                def normal(self, mean, std): return random.gauss(mean, std)
            return MockRandom()
    np = MockNumpy()


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    PRIVACY_VIOLATION = "privacy_violation"
    INTEGRITY_VIOLATION = "integrity_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    ENCRYPTION_FAILURE = "encryption_failure"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class SecurityAction(Enum):
    """Security response actions."""
    LOG = "log"
    ALERT = "alert"
    THROTTLE = "throttle"
    BLOCK = "block"
    ISOLATE = "isolate"
    ENCRYPT = "encrypt"
    AUDIT = "audit"


@dataclass
class ThreatIntelligence:
    """Threat intelligence data."""
    threat_id: str
    threat_type: str
    indicators: List[str]
    risk_score: float
    confidence: float
    source: str
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    source_component: str
    description: str
    timestamp: float
    context: Dict[str, Any]
    indicators: List[str]
    actions_taken: List[SecurityAction]
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.indicators is None:
            self.indicators = []
        if self.actions_taken is None:
            self.actions_taken = []


@dataclass
class AnomalyProfile:
    """Anomaly detection profile."""
    component: str
    baseline_metrics: Dict[str, float]
    thresholds: Dict[str, float]
    model_parameters: Dict[str, Any]
    last_updated: float
    samples_count: int


class AnomalyDetector:
    """ML-powered anomaly detection system."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.profiles: Dict[str, AnomalyProfile] = {}
        self.detection_history: List[Dict[str, Any]] = []
        self.learning_window = 1000  # Number of samples for learning
        
        self.logger = logging.getLogger(__name__)
    
    def learn_baseline(
        self,
        component: str,
        metrics: Dict[str, float],
        update_existing: bool = True
    ):
        """Learn baseline behavior for a component.
        
        Args:
            component: Component name
            metrics: Current metrics
            update_existing: Whether to update existing profile
        """
        if component not in self.profiles:
            # Create new profile
            self.profiles[component] = AnomalyProfile(
                component=component,
                baseline_metrics=metrics.copy(),
                thresholds={},
                model_parameters={'samples': [metrics]},
                last_updated=time.time(),
                samples_count=1
            )
        elif update_existing:
            # Update existing profile
            profile = self.profiles[component]
            samples = profile.model_parameters.get('samples', [])
            samples.append(metrics)
            
            # Keep only recent samples
            if len(samples) > self.learning_window:
                samples = samples[-self.learning_window:]
            
            # Recalculate baseline
            profile.baseline_metrics = self._calculate_baseline(samples)
            profile.thresholds = self._calculate_thresholds(samples)
            profile.model_parameters['samples'] = samples
            profile.last_updated = time.time()
            profile.samples_count += 1
    
    def detect_anomalies(
        self,
        component: str,
        current_metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in component metrics.
        
        Args:
            component: Component name
            current_metrics: Current metrics to analyze
            
        Returns:
            List of detected anomalies
        """
        if component not in self.profiles:
            self.logger.warning(f"No profile found for component: {component}")
            return []
        
        profile = self.profiles[component]
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in profile.baseline_metrics:
                continue
            
            baseline_value = profile.baseline_metrics[metric_name]
            threshold = profile.thresholds.get(metric_name, 0.0)
            
            # Calculate deviation
            if baseline_value != 0:
                deviation = abs(current_value - baseline_value) / abs(baseline_value)
            else:
                deviation = abs(current_value)
            
            # Check for anomaly
            if deviation > threshold:
                anomaly_score = min(deviation / threshold, 10.0)  # Cap at 10x
                
                anomaly = {
                    'component': component,
                    'metric': metric_name,
                    'current_value': current_value,
                    'baseline_value': baseline_value,
                    'deviation': deviation,
                    'threshold': threshold,
                    'anomaly_score': anomaly_score,
                    'severity': self._calculate_severity(anomaly_score),
                    'timestamp': time.time()
                }
                
                anomalies.append(anomaly)
                self.logger.warning(
                    f"Anomaly detected in {component}.{metric_name}: "
                    f"{current_value} vs baseline {baseline_value} "
                    f"(score: {anomaly_score:.2f})"
                )
        
        # Record detection
        if anomalies:
            self.detection_history.append({
                'component': component,
                'timestamp': time.time(),
                'anomalies': anomalies,
                'total_metrics': len(current_metrics)
            })
        
        # Learn from current metrics
        self.learn_baseline(component, current_metrics, update_existing=True)
        
        return anomalies
    
    def _calculate_baseline(self, samples: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate baseline metrics from samples.
        
        Args:
            samples: List of metric samples
            
        Returns:
            Baseline metrics
        """
        if not samples:
            return {}
        
        baseline = {}
        
        # Get all metric names
        all_metrics = set()
        for sample in samples:
            all_metrics.update(sample.keys())
        
        # Calculate baseline for each metric
        for metric in all_metrics:
            values = [sample.get(metric, 0) for sample in samples if metric in sample]
            if values:
                baseline[metric] = np.mean(values)
        
        return baseline
    
    def _calculate_thresholds(self, samples: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate anomaly thresholds from samples.
        
        Args:
            samples: List of metric samples
            
        Returns:
            Threshold values
        """
        if len(samples) < 10:
            # Use default thresholds for small sample sizes
            return {metric: 0.5 for sample in samples for metric in sample.keys()}
        
        thresholds = {}
        
        # Get all metric names
        all_metrics = set()
        for sample in samples:
            all_metrics.update(sample.keys())
        
        # Calculate threshold for each metric (based on std deviation)
        for metric in all_metrics:
            values = [sample.get(metric, 0) for sample in samples if metric in sample]
            if values and len(values) >= 5:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                
                # Use coefficient of variation as base threshold
                if mean_val != 0:
                    threshold = min(max(std_dev / abs(mean_val), 0.1), 2.0)  # Between 10% and 200%
                else:
                    threshold = 0.5
                
                thresholds[metric] = threshold
        
        return thresholds
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate severity from anomaly score.
        
        Args:
            anomaly_score: Anomaly score
            
        Returns:
            Severity level
        """
        if anomaly_score >= 5.0:
            return "critical"
        elif anomaly_score >= 3.0:
            return "high"
        elif anomaly_score >= 2.0:
            return "medium"
        else:
            return "low"


class ThreatIntelligenceEngine:
    """Threat intelligence collection and analysis."""
    
    def __init__(self):
        """Initialize threat intelligence engine."""
        self.threat_feeds: Dict[str, ThreatIntelligence] = {}
        self.ioc_database: Dict[str, List[str]] = {}  # Indicators of Compromise
        self.risk_scores: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
        self._initialize_threat_data()
    
    def _initialize_threat_data(self):
        """Initialize with basic threat intelligence."""
        # Known malicious patterns
        self.add_threat_intelligence(ThreatIntelligence(
            threat_id="malicious_patterns_001",
            threat_type="injection_attack",
            indicators=["<script>", "javascript:", "eval(", "exec("],
            risk_score=0.9,
            confidence=0.95,
            source="builtin_rules",
            timestamp=time.time(),
            metadata={"category": "web_attacks"}
        ))
        
        self.add_threat_intelligence(ThreatIntelligence(
            threat_id="data_exfiltration_001",
            threat_type="data_breach",
            indicators=["SELECT * FROM", "UNION SELECT", "DROP TABLE"],
            risk_score=0.95,
            confidence=0.9,
            source="builtin_rules",
            timestamp=time.time(),
            metadata={"category": "sql_injection"}
        ))
        
        # Suspicious behavior patterns
        self.add_threat_intelligence(ThreatIntelligence(
            threat_id="anomalous_access_001",
            threat_type="unauthorized_access",
            indicators=["rapid_requests", "unusual_timestamps", "geographic_anomaly"],
            risk_score=0.7,
            confidence=0.8,
            source="behavioral_analysis",
            timestamp=time.time(),
            metadata={"category": "access_patterns"}
        ))
    
    def add_threat_intelligence(self, threat: ThreatIntelligence):
        """Add threat intelligence data.
        
        Args:
            threat: Threat intelligence to add
        """
        self.threat_feeds[threat.threat_id] = threat
        
        # Update IOC database
        for indicator in threat.indicators:
            if threat.threat_type not in self.ioc_database:
                self.ioc_database[threat.threat_type] = []
            
            if indicator not in self.ioc_database[threat.threat_type]:
                self.ioc_database[threat.threat_type].append(indicator)
        
        # Update risk scores
        self.risk_scores[threat.threat_type] = max(
            self.risk_scores.get(threat.threat_type, 0),
            threat.risk_score
        )
        
        self.logger.info(f"Added threat intelligence: {threat.threat_id}")
    
    def assess_threat(
        self,
        data: str,
        context: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Assess threat level of data.
        
        Args:
            data: Data to assess
            context: Context information
            
        Returns:
            Tuple of (risk_score, matched_indicators)
        """
        max_risk = 0.0
        matched_indicators = []
        
        data_lower = data.lower()
        
        for threat_id, threat in self.threat_feeds.items():
            for indicator in threat.indicators:
                if indicator.lower() in data_lower:
                    max_risk = max(max_risk, threat.risk_score)
                    matched_indicators.append(f"{threat.threat_type}:{indicator}")
        
        # Contextual risk assessment
        contextual_risk = self._assess_contextual_risk(context)
        combined_risk = min(max_risk + contextual_risk * 0.3, 1.0)
        
        return combined_risk, matched_indicators
    
    def _assess_contextual_risk(self, context: Dict[str, Any]) -> float:
        """Assess contextual risk factors.
        
        Args:
            context: Context information
            
        Returns:
            Contextual risk score (0-1)
        """
        risk_factors = 0.0
        
        # Time-based risk
        current_hour = time.localtime().tm_hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            risk_factors += 0.2
        
        # Frequency-based risk
        request_count = context.get('request_count_last_hour', 0)
        if request_count > 100:
            risk_factors += min(request_count / 1000, 0.5)
        
        # Geographic risk (simulated)
        source_country = context.get('source_country', 'unknown')
        high_risk_countries = ['unknown', 'tor_exit', 'suspicious']
        if source_country in high_risk_countries:
            risk_factors += 0.3
        
        return min(risk_factors, 1.0)


class AdaptiveSecurityControls:
    """Adaptive security controls that adjust based on threat level."""
    
    def __init__(self):
        """Initialize adaptive security controls."""
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        self.active_controls: Dict[str, Dict[str, Any]] = {}
        self.control_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        # Rate limiting policy
        self.security_policies['rate_limiting'] = {
            'low_threat': {'max_requests_per_minute': 100, 'burst_size': 20},
            'medium_threat': {'max_requests_per_minute': 50, 'burst_size': 10},
            'high_threat': {'max_requests_per_minute': 20, 'burst_size': 5},
            'critical_threat': {'max_requests_per_minute': 5, 'burst_size': 1}
        }
        
        # Encryption policy
        self.security_policies['encryption'] = {
            'low_threat': {'algorithm': 'AES-256', 'key_rotation_hours': 24},
            'medium_threat': {'algorithm': 'AES-256', 'key_rotation_hours': 12},
            'high_threat': {'algorithm': 'ChaCha20-Poly1305', 'key_rotation_hours': 6},
            'critical_threat': {'algorithm': 'ChaCha20-Poly1305', 'key_rotation_hours': 1}
        }
        
        # Access control policy
        self.security_policies['access_control'] = {
            'low_threat': {'require_mfa': False, 'session_timeout_minutes': 60},
            'medium_threat': {'require_mfa': True, 'session_timeout_minutes': 30},
            'high_threat': {'require_mfa': True, 'session_timeout_minutes': 15},
            'critical_threat': {'require_mfa': True, 'session_timeout_minutes': 5}
        }
        
        # Audit policy
        self.security_policies['audit'] = {
            'low_threat': {'log_level': 'info', 'retention_days': 30},
            'medium_threat': {'log_level': 'debug', 'retention_days': 60},
            'high_threat': {'log_level': 'trace', 'retention_days': 90},
            'critical_threat': {'log_level': 'trace', 'retention_days': 365}
        }
    
    def adapt_controls(
        self,
        component: str,
        threat_level: ThreatLevel,
        incident_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Adapt security controls based on threat level.
        
        Args:
            component: Component name
            threat_level: Current threat level
            incident_context: Context from security incident
            
        Returns:
            Applied security controls
        """
        controls_applied = {}
        threat_key = threat_level.value
        
        for policy_name, policy_config in self.security_policies.items():
            if threat_key in policy_config:
                control_config = policy_config[threat_key].copy()
                
                # Apply contextual modifications
                if incident_context:
                    control_config = self._modify_controls_for_context(
                        control_config, policy_name, incident_context
                    )
                
                controls_applied[policy_name] = control_config
                
                # Store active control
                control_key = f"{component}:{policy_name}"
                self.active_controls[control_key] = {
                    'component': component,
                    'policy': policy_name,
                    'config': control_config,
                    'threat_level': threat_level.value,
                    'applied_at': time.time(),
                    'context': incident_context or {}
                }
        
        # Record control change
        self.control_history.append({
            'component': component,
            'threat_level': threat_level.value,
            'controls_applied': controls_applied,
            'timestamp': time.time(),
            'context': incident_context or {}
        })
        
        self.logger.info(
            f"Adapted security controls for {component} "
            f"(threat level: {threat_level.value}): {list(controls_applied.keys())}"
        )
        
        return controls_applied
    
    def _modify_controls_for_context(
        self,
        control_config: Dict[str, Any],
        policy_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modify controls based on incident context.
        
        Args:
            control_config: Base control configuration
            policy_name: Policy name
            context: Incident context
            
        Returns:
            Modified control configuration
        """
        modified_config = control_config.copy()
        
        # Rate limiting modifications
        if policy_name == 'rate_limiting':
            if context.get('attack_type') == 'ddos':
                modified_config['max_requests_per_minute'] //= 2
                modified_config['burst_size'] = 1
            elif context.get('repeated_offender'):
                modified_config['max_requests_per_minute'] //= 3
        
        # Encryption modifications
        elif policy_name == 'encryption':
            if context.get('data_sensitivity') == 'high':
                modified_config['key_rotation_hours'] //= 2
        
        # Access control modifications
        elif policy_name == 'access_control':
            if context.get('privilege_escalation_attempt'):
                modified_config['session_timeout_minutes'] //= 2
                modified_config['require_additional_auth'] = True
        
        return modified_config
    
    def get_active_controls(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get currently active security controls.
        
        Args:
            component: Optional component filter
            
        Returns:
            Active controls dictionary
        """
        if component:
            return {
                k: v for k, v in self.active_controls.items()
                if v['component'] == component
            }
        
        return self.active_controls.copy()


class AdaptiveSecuritySystem:
    """Comprehensive adaptive security system."""
    
    def __init__(self):
        """Initialize adaptive security system."""
        self.anomaly_detector = AnomalyDetector()
        self.threat_intelligence = ThreatIntelligenceEngine()
        self.security_controls = AdaptiveSecurityControls()
        
        self.incidents: Dict[str, SecurityIncident] = {}
        self.security_metrics: Dict[str, List[float]] = {}
        self.monitoring_active = False
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_security_event(
        self,
        component: str,
        event_data: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> Optional[SecurityIncident]:
        """Analyze a security event and determine response.
        
        Args:
            component: Component generating the event
            event_data: Event data to analyze
            metrics: Optional component metrics
            
        Returns:
            Security incident if threat detected, None otherwise
        """
        # Anomaly detection
        anomalies = []
        if metrics:
            anomalies = self.anomaly_detector.detect_anomalies(component, metrics)
        
        # Threat intelligence assessment
        event_text = json.dumps(event_data)
        risk_score, indicators = self.threat_intelligence.assess_threat(
            event_text, event_data
        )
        
        # Determine threat level
        threat_level = self._calculate_threat_level(risk_score, anomalies)
        
        if threat_level != ThreatLevel.LOW or anomalies:
            # Create security incident
            incident = await self._create_security_incident(
                component, event_data, threat_level, risk_score, indicators, anomalies
            )
            
            # Apply adaptive controls
            await self._respond_to_incident(incident)
            
            return incident
        
        return None
    
    def _calculate_threat_level(
        self,
        risk_score: float,
        anomalies: List[Dict[str, Any]]
    ) -> ThreatLevel:
        """Calculate overall threat level.
        
        Args:
            risk_score: Threat intelligence risk score
            anomalies: Detected anomalies
            
        Returns:
            Overall threat level
        """
        # Base threat from risk score
        if risk_score >= 0.9:
            base_threat = ThreatLevel.CRITICAL
        elif risk_score >= 0.7:
            base_threat = ThreatLevel.HIGH
        elif risk_score >= 0.5:
            base_threat = ThreatLevel.MEDIUM
        else:
            base_threat = ThreatLevel.LOW
        
        # Adjust based on anomalies
        if anomalies:
            critical_anomalies = [a for a in anomalies if a['severity'] == 'critical']
            high_anomalies = [a for a in anomalies if a['severity'] == 'high']
            
            if critical_anomalies:
                if base_threat in [ThreatLevel.LOW, ThreatLevel.MEDIUM]:
                    return ThreatLevel.HIGH
                elif base_threat == ThreatLevel.HIGH:
                    return ThreatLevel.CRITICAL
            elif high_anomalies and base_threat == ThreatLevel.LOW:
                return ThreatLevel.MEDIUM
        
        return base_threat
    
    async def _create_security_incident(
        self,
        component: str,
        event_data: Dict[str, Any],
        threat_level: ThreatLevel,
        risk_score: float,
        indicators: List[str],
        anomalies: List[Dict[str, Any]]
    ) -> SecurityIncident:
        """Create a security incident record.
        
        Args:
            component: Source component
            event_data: Event data
            threat_level: Calculated threat level
            risk_score: Risk score from threat intelligence
            indicators: Matched threat indicators
            anomalies: Detected anomalies
            
        Returns:
            Created security incident
        """
        incident_id = str(uuid.uuid4())
        
        # Determine event type
        event_type = self._classify_event_type(indicators, anomalies)
        
        # Create description
        description = f"Security incident in {component}: "
        if indicators:
            description += f"Threat indicators: {', '.join(indicators[:3])}. "
        if anomalies:
            description += f"Anomalies detected: {len(anomalies)} metrics. "
        description += f"Risk score: {risk_score:.2f}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            event_type=event_type,
            threat_level=threat_level,
            source_component=component,
            description=description,
            timestamp=time.time(),
            context={
                'event_data': event_data,
                'risk_score': risk_score,
                'anomalies': anomalies,
                'threat_intelligence_version': time.time()
            },
            indicators=indicators,
            actions_taken=[]
        )
        
        self.incidents[incident_id] = incident
        self.logger.warning(f"Created security incident {incident_id}: {description}")
        
        return incident
    
    def _classify_event_type(
        self,
        indicators: List[str],
        anomalies: List[Dict[str, Any]]
    ) -> SecurityEvent:
        """Classify the type of security event.
        
        Args:
            indicators: Threat indicators
            anomalies: Detected anomalies
            
        Returns:
            Classified security event type
        """
        # Check threat indicators
        for indicator in indicators:
            if 'injection_attack' in indicator or 'data_breach' in indicator:
                return SecurityEvent.DATA_BREACH
            elif 'unauthorized_access' in indicator:
                return SecurityEvent.UNAUTHORIZED_ACCESS
        
        # Check anomaly patterns
        if anomalies:
            privacy_anomalies = [a for a in anomalies if 'privacy' in a['metric'].lower()]
            if privacy_anomalies:
                return SecurityEvent.PRIVACY_VIOLATION
            
            integrity_anomalies = [a for a in anomalies if 'integrity' in a['metric'].lower()]
            if integrity_anomalies:
                return SecurityEvent.INTEGRITY_VIOLATION
        
        return SecurityEvent.ANOMALOUS_BEHAVIOR
    
    async def _respond_to_incident(self, incident: SecurityIncident):
        """Respond to a security incident.
        
        Args:
            incident: Security incident to respond to
        """
        # Adapt security controls
        controls_applied = self.security_controls.adapt_controls(
            incident.source_component,
            incident.threat_level,
            incident.context
        )
        
        # Determine and execute actions
        actions = self._determine_response_actions(incident, controls_applied)
        
        for action in actions:
            await self._execute_security_action(action, incident)
            incident.actions_taken.append(action)
        
        self.logger.info(
            f"Responded to incident {incident.incident_id} with actions: "
            f"{[action.value for action in actions]}"
        )
    
    def _determine_response_actions(
        self,
        incident: SecurityIncident,
        controls_applied: Dict[str, Any]
    ) -> List[SecurityAction]:
        """Determine appropriate response actions.
        
        Args:
            incident: Security incident
            controls_applied: Applied security controls
            
        Returns:
            List of security actions to take
        """
        actions = [SecurityAction.LOG, SecurityAction.AUDIT]  # Always log and audit
        
        if incident.threat_level == ThreatLevel.CRITICAL:
            actions.extend([
                SecurityAction.BLOCK,
                SecurityAction.ISOLATE,
                SecurityAction.ALERT,
                SecurityAction.ENCRYPT
            ])
        elif incident.threat_level == ThreatLevel.HIGH:
            actions.extend([
                SecurityAction.THROTTLE,
                SecurityAction.ALERT,
                SecurityAction.ENCRYPT
            ])
        elif incident.threat_level == ThreatLevel.MEDIUM:
            actions.extend([
                SecurityAction.THROTTLE,
                SecurityAction.ALERT
            ])
        
        # Event-specific actions
        if incident.event_type == SecurityEvent.DATA_BREACH:
            actions.append(SecurityAction.ENCRYPT)
        elif incident.event_type == SecurityEvent.RATE_LIMIT_EXCEEDED:
            actions.append(SecurityAction.THROTTLE)
        
        return list(set(actions))  # Remove duplicates
    
    async def _execute_security_action(
        self,
        action: SecurityAction,
        incident: SecurityIncident
    ):
        """Execute a security action.
        
        Args:
            action: Security action to execute
            incident: Related security incident
        """
        try:
            if action == SecurityAction.LOG:
                self.logger.warning(f"Security incident: {incident.description}")
            
            elif action == SecurityAction.ALERT:
                await self._send_security_alert(incident)
            
            elif action == SecurityAction.THROTTLE:
                await self._apply_throttling(incident.source_component)
            
            elif action == SecurityAction.BLOCK:
                await self._block_component(incident.source_component)
            
            elif action == SecurityAction.ISOLATE:
                await self._isolate_component(incident.source_component)
            
            elif action == SecurityAction.ENCRYPT:
                await self._enhance_encryption(incident.source_component)
            
            elif action == SecurityAction.AUDIT:
                await self._trigger_audit(incident)
            
        except Exception as e:
            self.logger.error(f"Error executing security action {action.value}: {e}")
    
    async def _send_security_alert(self, incident: SecurityIncident):
        """Send security alert."""
        alert_data = {
            'incident_id': incident.incident_id,
            'threat_level': incident.threat_level.value,
            'component': incident.source_component,
            'description': incident.description,
            'timestamp': incident.timestamp
        }
        
        # In real implementation, this would send to SIEM, email, etc.
        self.logger.critical(f"SECURITY ALERT: {json.dumps(alert_data)}")
    
    async def _apply_throttling(self, component: str):
        """Apply throttling to component."""
        self.logger.info(f"Applied throttling to component: {component}")
        # Implementation would integrate with actual throttling mechanism
    
    async def _block_component(self, component: str):
        """Block component access."""
        self.logger.warning(f"Blocked component: {component}")
        # Implementation would integrate with access control system
    
    async def _isolate_component(self, component: str):
        """Isolate component."""
        self.logger.critical(f"Isolated component: {component}")
        # Implementation would isolate component from network/system
    
    async def _enhance_encryption(self, component: str):
        """Enhance encryption for component."""
        self.logger.info(f"Enhanced encryption for component: {component}")
        # Implementation would upgrade encryption parameters
    
    async def _trigger_audit(self, incident: SecurityIncident):
        """Trigger comprehensive audit."""
        audit_data = {
            'incident_id': incident.incident_id,
            'component': incident.source_component,
            'timestamp': incident.timestamp,
            'context': incident.context
        }
        
        self.logger.info(f"Triggered security audit: {json.dumps(audit_data)}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data.
        
        Returns:
            Security dashboard information
        """
        current_time = time.time()
        recent_incidents = [
            inc for inc in self.incidents.values()
            if current_time - inc.timestamp < 3600  # Last hour
        ]
        
        # Threat level distribution
        threat_distribution = {
            level.value: len([
                inc for inc in recent_incidents
                if inc.threat_level == level
            ])
            for level in ThreatLevel
        }
        
        # Active controls
        active_controls = self.security_controls.get_active_controls()
        
        return {
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'total_incidents': len(self.incidents),
            'recent_incidents': len(recent_incidents),
            'unresolved_incidents': len([
                inc for inc in self.incidents.values()
                if not inc.resolved
            ]),
            'threat_distribution': threat_distribution,
            'active_controls': len(active_controls),
            'anomaly_profiles': len(self.anomaly_detector.profiles),
            'threat_intelligence_feeds': len(self.threat_intelligence.threat_feeds),
            'security_score': self._calculate_security_score()
        }
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-1).
        
        Returns:
            Security score
        """
        # Base score
        score = 1.0
        
        # Deduct for unresolved critical incidents
        critical_incidents = [
            inc for inc in self.incidents.values()
            if not inc.resolved and inc.threat_level == ThreatLevel.CRITICAL
        ]
        score -= min(len(critical_incidents) * 0.2, 0.8)
        
        # Deduct for high threat levels
        high_incidents = [
            inc for inc in self.incidents.values()
            if not inc.resolved and inc.threat_level == ThreatLevel.HIGH
        ]
        score -= min(len(high_incidents) * 0.1, 0.5)
        
        # Bonus for active monitoring and controls
        if self.monitoring_active:
            score += 0.1
        
        if self.security_controls.get_active_controls():
            score += 0.05
        
        return max(min(score, 1.0), 0.0)
    
    def export_security_report(self, output_path: Path) -> None:
        """Export comprehensive security report.
        
        Args:
            output_path: Path to save report
        """
        report = {
            'dashboard': self.get_security_dashboard(),
            'incidents': {k: asdict(v) for k, v in self.incidents.items()},
            'anomaly_profiles': {
                k: asdict(v) for k, v in self.anomaly_detector.profiles.items()
            },
            'threat_intelligence': {
                k: asdict(v) for k, v in self.threat_intelligence.threat_feeds.items()
            },
            'active_controls': self.security_controls.get_active_controls(),
            'generated_at': time.time()
        }
        
        output_path.write_text(json.dumps(report, indent=2, default=str))
        self.logger.info(f"Security report exported to {output_path}")