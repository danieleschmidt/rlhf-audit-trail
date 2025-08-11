"""
Advanced security hardening and protection mechanisms for RLHF audit trail.
Generation 2: Security measures, access control, and threat protection.
"""

import hashlib
import hmac
import secrets
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import re
import ipaddress

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    import jwt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from .exceptions import SecurityError, AuthenticationError, AuthorizationError


logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security enforcement levels."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class AccessLevel(Enum):
    """User access levels."""
    READ_ONLY = "read_only"
    ANNOTATOR = "annotator"
    TRAINER = "trainer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security-related event."""
    event_type: str
    severity: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    blocked: bool = False


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int
    burst_allowance: int
    window_size_seconds: int = 60


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    # Dangerous patterns that should be blocked
    BLOCKED_PATTERNS = [
        # SQL injection patterns
        r"(?i)(union\s+select|insert\s+into|delete\s+from|drop\s+table)",
        r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
        r"(?i)(exec\s*\(|execute\s*\()",
        r"(?i)(sp_|xp_)\w+",
        
        # Script injection patterns  
        r"<script[^>]*>.*?</script>",
        r"javascript:\s*",
        r"on\w+\s*=\s*['\"][^'\"]*['\"]",
        r"eval\s*\(",
        r"setTimeout\s*\(",
        r"setInterval\s*\(",
        
        # Command injection patterns
        r"[;&|`$]\s*\w+",
        r"\|\s*(cat|ls|pwd|whoami|id)",
        r"&&\s*(rm|mv|cp|chmod)",
        
        # Path traversal patterns
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        
        # Template injection patterns
        r"\{\{\s*.*\s*\}\}",
        r"\{\%\s*.*\s*\%\}",
        
        # LDAP injection patterns
        r"[()=*<>~!&|]",
        
        # XML injection patterns
        r"<!\[CDATA\[",
        r"<!ENTITY",
        r"<!DOCTYPE",
        
        # NoSQL injection patterns
        r"\$where\s*:",
        r"\$ne\s*:",
        r"\$gt\s*:",
        r"\$regex\s*:"
    ]
    
    # Allowed characters for different input types
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise SecurityError("Input must be a string")
        
        # Check length
        if len(value) > max_length:
            raise SecurityError(f"Input exceeds maximum length of {max_length}")
        
        # Check for dangerous patterns
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE | re.MULTILINE):
                raise SecurityError(f"Input contains potentially dangerous pattern: {pattern}")
        
        # Remove or encode HTML if not allowed
        if not allow_html:
            value = cls._encode_html(value)
        
        # Remove null bytes and control characters
        value = cls._remove_dangerous_chars(value)
        
        return value.strip()
    
    @classmethod
    def sanitize_email(cls, email: str) -> str:
        """Sanitize email address."""
        email = email.lower().strip()
        
        if not cls.EMAIL_PATTERN.match(email):
            raise SecurityError("Invalid email format")
        
        if len(email) > 254:  # RFC 5321 limit
            raise SecurityError("Email address too long")
        
        return email
    
    @classmethod
    def sanitize_uuid(cls, uuid_str: str) -> str:
        """Sanitize UUID string."""
        uuid_str = uuid_str.lower().strip()
        
        if not cls.UUID_PATTERN.match(uuid_str):
            raise SecurityError("Invalid UUID format")
        
        return uuid_str
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename for safe file operations."""
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Check for dangerous names
        dangerous_names = [
            'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
            'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3',
            'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
        ]
        
        if filename.lower() in dangerous_names:
            raise SecurityError("Filename not allowed")
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename
    
    @classmethod
    def _encode_html(cls, text: str) -> str:
        """Encode HTML special characters."""
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        return "".join(html_escape_table.get(c, c) for c in text)
    
    @classmethod
    def _remove_dangerous_chars(cls, text: str) -> str:
        """Remove null bytes and control characters."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove other dangerous control characters but keep common whitespace
        allowed_control_chars = {'\t', '\n', '\r'}
        text = ''.join(c for c in text if ord(c) >= 32 or c in allowed_control_chars)
        
        return text


class RateLimiter:
    """Rate limiting for API endpoints and operations."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limits."""
        current_time = time.time()
        
        with self.lock:
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            request_times = self.requests[identifier]
            
            # Remove old requests outside the window
            window_start = current_time - self.config.window_size_seconds
            request_times[:] = [t for t in request_times if t > window_start]
            
            # Check rate limit
            if len(request_times) >= self.config.requests_per_minute:
                return False, {
                    "allowed": False,
                    "reason": "rate_limit_exceeded",
                    "reset_time": request_times[0] + self.config.window_size_seconds,
                    "requests_remaining": 0
                }
            
            # Add current request
            request_times.append(current_time)
            
            return True, {
                "allowed": True,
                "requests_remaining": self.config.requests_per_minute - len(request_times),
                "reset_time": current_time + self.config.window_size_seconds
            }
    
    def cleanup_old_entries(self):
        """Clean up old rate limit entries."""
        current_time = time.time()
        window_start = current_time - self.config.window_size_seconds
        
        with self.lock:
            to_remove = []
            for identifier, request_times in self.requests.items():
                # Remove old requests
                request_times[:] = [t for t in request_times if t > window_start]
                
                # Mark empty entries for removal
                if not request_times:
                    to_remove.append(identifier)
            
            for identifier in to_remove:
                del self.requests[identifier]


class AccessController:
    """Role-based access control system."""
    
    def __init__(self):
        self.user_roles: Dict[str, AccessLevel] = {}
        self.role_permissions: Dict[AccessLevel, Set[str]] = {
            AccessLevel.READ_ONLY: {"read_audit_logs", "read_model_cards"},
            AccessLevel.ANNOTATOR: {"read_audit_logs", "create_annotations", "read_model_cards"},
            AccessLevel.TRAINER: {
                "read_audit_logs", "create_annotations", "track_policy_updates",
                "create_checkpoints", "read_model_cards", "generate_model_cards"
            },
            AccessLevel.ADMIN: {
                "read_audit_logs", "create_annotations", "track_policy_updates",
                "create_checkpoints", "read_model_cards", "generate_model_cards",
                "manage_sessions", "export_data", "view_system_status"
            },
            AccessLevel.SUPER_ADMIN: {
                "read_audit_logs", "create_annotations", "track_policy_updates",
                "create_checkpoints", "read_model_cards", "generate_model_cards",
                "manage_sessions", "export_data", "view_system_status",
                "manage_users", "manage_system", "delete_data"
            }
        }
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
    
    def authenticate_user(self, user_id: str, password: str, ip_address: str) -> Optional[str]:
        """Authenticate user and return session token."""
        # Check if IP is blocked
        if self._is_ip_blocked(ip_address):
            raise SecurityError("IP address is temporarily blocked")
        
        # Check rate limiting for authentication attempts
        if not self._check_auth_rate_limit(ip_address):
            raise SecurityError("Too many authentication attempts")
        
        # TODO: Implement actual password verification
        # This is a placeholder - in production, use proper password hashing
        if self._verify_password(user_id, password):
            # Generate session token
            session_token = self._generate_session_token(user_id)
            
            # Clear failed attempts
            if ip_address in self.failed_attempts:
                del self.failed_attempts[ip_address]
            
            return session_token
        else:
            # Record failed attempt
            self._record_failed_attempt(ip_address)
            raise AuthenticationError("Invalid credentials")
    
    def authorize_action(self, session_token: str, action: str) -> bool:
        """Check if user is authorized to perform action."""
        session = self.session_tokens.get(session_token)
        if not session:
            return False
        
        # Check session expiry
        if session["expires"] < datetime.utcnow():
            del self.session_tokens[session_token]
            return False
        
        user_id = session["user_id"]
        user_role = self.user_roles.get(user_id)
        
        if not user_role:
            return False
        
        permissions = self.role_permissions.get(user_role, set())
        return action in permissions
    
    def revoke_session(self, session_token: str):
        """Revoke a session token."""
        if session_token in self.session_tokens:
            del self.session_tokens[session_token]
    
    def add_user(self, user_id: str, role: AccessLevel):
        """Add a user with specified role."""
        self.user_roles[user_id] = role
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify user password (placeholder implementation)."""
        # TODO: Implement proper password verification with hashing
        # This is a placeholder for demonstration
        return len(password) >= 8  # Simple check for demo
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate secure session token."""
        token = secrets.token_urlsafe(32)
        expires = datetime.utcnow() + timedelta(hours=8)  # 8-hour sessions
        
        self.session_tokens[token] = {
            "user_id": user_id,
            "created": datetime.utcnow(),
            "expires": expires,
            "last_activity": datetime.utcnow()
        }
        
        return token
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        if ip_address in self.blocked_ips:
            block_time = self.blocked_ips[ip_address]
            if datetime.utcnow() < block_time:
                return True
            else:
                # Block expired
                del self.blocked_ips[ip_address]
        return False
    
    def _check_auth_rate_limit(self, ip_address: str) -> bool:
        """Check authentication rate limit."""
        current_time = time.time()
        window_size = 300  # 5 minutes
        max_attempts = 5
        
        if ip_address not in self.failed_attempts:
            return True
        
        # Clean old attempts
        attempts = self.failed_attempts[ip_address]
        attempts[:] = [t for t in attempts if t > current_time - window_size]
        
        return len(attempts) < max_attempts
    
    def _record_failed_attempt(self, ip_address: str):
        """Record failed authentication attempt."""
        current_time = time.time()
        
        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = []
        
        self.failed_attempts[ip_address].append(current_time)
        
        # Block IP if too many failures
        if len(self.failed_attempts[ip_address]) >= 10:  # 10 failures
            self.blocked_ips[ip_address] = datetime.utcnow() + timedelta(hours=1)
            logger.warning(f"Blocked IP {ip_address} due to excessive failed attempts")


class SecurityMonitor:
    """Security event monitoring and threat detection."""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.threat_patterns: Dict[str, List[str]] = {
            "brute_force": [
                r"multiple failed login attempts",
                r"authentication rate limit exceeded"
            ],
            "injection_attack": [
                r"SQL injection detected",
                r"script injection detected",
                r"command injection detected"
            ],
            "unauthorized_access": [
                r"unauthorized action attempted",
                r"invalid session token",
                r"privilege escalation attempt"
            ]
        }
        self.ip_reputation: Dict[str, ThreatLevel] = {}
    
    def log_security_event(
        self,
        event_type: str,
        severity: ThreatLevel,
        details: Dict[str, Any],
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        blocked: bool = False
    ):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            details=details,
            blocked=blocked
        )
        
        self.security_events.append(event)
        
        # Update IP reputation
        if source_ip and severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._update_ip_reputation(source_ip, severity)
        
        # Log the event
        logger.warning(f"Security event: {event_type} from {source_ip or 'unknown'} - {details}")
        
        # Trigger automated responses for critical events
        if severity == ThreatLevel.CRITICAL:
            self._handle_critical_event(event)
    
    def analyze_threats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze recent security events for threat patterns."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        analysis = {
            "total_events": len(recent_events),
            "events_by_severity": {},
            "events_by_type": {},
            "top_source_ips": {},
            "threat_indicators": []
        }
        
        # Count by severity
        for severity in ThreatLevel:
            count = sum(1 for e in recent_events if e.severity == severity)
            analysis["events_by_severity"][severity.value] = count
        
        # Count by type
        event_types = {}
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        analysis["events_by_type"] = event_types
        
        # Top source IPs
        ip_counts = {}
        for event in recent_events:
            if event.source_ip:
                ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        analysis["top_source_ips"] = dict(sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Detect threat patterns
        analysis["threat_indicators"] = self._detect_threat_patterns(recent_events)
        
        return analysis
    
    def get_ip_reputation(self, ip_address: str) -> ThreatLevel:
        """Get reputation level for IP address."""
        return self.ip_reputation.get(ip_address, ThreatLevel.LOW)
    
    def _update_ip_reputation(self, ip_address: str, severity: ThreatLevel):
        """Update IP reputation based on security events."""
        current_level = self.ip_reputation.get(ip_address, ThreatLevel.LOW)
        
        # Escalate reputation based on severity
        if severity == ThreatLevel.CRITICAL:
            self.ip_reputation[ip_address] = ThreatLevel.CRITICAL
        elif severity == ThreatLevel.HIGH and current_level != ThreatLevel.CRITICAL:
            self.ip_reputation[ip_address] = ThreatLevel.HIGH
        elif current_level == ThreatLevel.LOW:
            self.ip_reputation[ip_address] = ThreatLevel.MEDIUM
    
    def _handle_critical_event(self, event: SecurityEvent):
        """Handle critical security events with automated responses."""
        if event.source_ip:
            # TODO: Implement automated IP blocking
            logger.critical(f"Critical security event from {event.source_ip} - automated blocking recommended")
        
        # TODO: Send alerts to security team
        # TODO: Trigger incident response procedures
    
    def _detect_threat_patterns(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Detect threat patterns in security events."""
        indicators = []
        
        # Group events by source IP
        events_by_ip = {}
        for event in events:
            if event.source_ip:
                if event.source_ip not in events_by_ip:
                    events_by_ip[event.source_ip] = []
                events_by_ip[event.source_ip].append(event)
        
        # Check for suspicious patterns
        for ip, ip_events in events_by_ip.items():
            # Multiple high-severity events from same IP
            high_severity_count = sum(1 for e in ip_events if e.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
            if high_severity_count >= 3:
                indicators.append({
                    "pattern": "multiple_high_severity_events",
                    "source_ip": ip,
                    "count": high_severity_count,
                    "risk_level": "high"
                })
            
            # Rapid succession of events (possible automated attack)
            if len(ip_events) >= 10:
                time_span = (ip_events[-1].timestamp - ip_events[0].timestamp).total_seconds()
                if time_span < 300:  # 10+ events in 5 minutes
                    indicators.append({
                        "pattern": "rapid_event_succession",
                        "source_ip": ip,
                        "events_per_minute": len(ip_events) / (time_span / 60),
                        "risk_level": "critical"
                    })
        
        return indicators


class EncryptionManager:
    """Advanced encryption and key management."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available - encryption disabled")
            return
        
        self.master_key = master_key or self._generate_master_key()
        self.cipher_suite = Fernet(self.master_key)
        self.key_rotation_interval = timedelta(days=30)
        self.last_key_rotation = datetime.utcnow()
    
    def encrypt_sensitive_data(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        if not CRYPTO_AVAILABLE:
            return data.encode('utf-8')
        
        return self.cipher_suite.encrypt(data.encode('utf-8'))
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        if not CRYPTO_AVAILABLE:
            return encrypted_data.decode('utf-8')
        
        return self.cipher_suite.decrypt(encrypted_data).decode('utf-8')
    
    def generate_secure_hash(self, data: str, salt: Optional[str] = None) -> str:
        """Generate secure hash with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        combined = f"{data}{salt}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify data against hash."""
        return self.generate_secure_hash(data, salt) == hash_value
    
    def _generate_master_key(self) -> bytes:
        """Generate a master encryption key."""
        if not CRYPTO_AVAILABLE:
            return b'dummy_key'
        
        return Fernet.generate_key()


# Global security components
_global_access_controller = AccessController()
_global_security_monitor = SecurityMonitor()
_global_rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=100, burst_allowance=10))
_global_encryption_manager = EncryptionManager()

def get_access_controller() -> AccessController:
    """Get global access controller."""
    return _global_access_controller

def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor."""
    return _global_security_monitor

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter."""
    return _global_rate_limiter

def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager."""
    return _global_encryption_manager