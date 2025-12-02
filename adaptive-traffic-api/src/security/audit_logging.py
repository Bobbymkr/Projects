"""
Advanced Security Audit Logging Framework.

Implements comprehensive security event logging:
- Authentication events
- Authorization failures
- Security violations
- Data access logs
- System changes
- Threat detection alerts
- Compliance reporting
"""

import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)

class SecurityEventType(Enum):
    """Security event classification."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_CONTROL = "access_control"
    DATA_ACCESS = "data_access"
    SECURITY_VIOLATION = "security_violation"
    THREAT_DETECTION = "threat_detection"
    SYSTEM_CHANGE = "system_change"
    AUDIT_TRAIL = "audit_trail"

class SecurityLevel(Enum):
    """Security event severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    timestamp: datetime
    event_type: SecurityEventType
    severity: SecurityLevel
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    action: str
    resource: Optional[str]
    outcome: str  # SUCCESS, FAILURE, BLOCKED
    details: Dict[str, Any]
    risk_score: float = 0.0
    threat_indicators: List[str] = None
    
    def __post_init__(self):
        if self.threat_indicators is None:
            self.threat_indicators = []

class SecurityAuditLogger:
    """Advanced security audit logging system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.event_buffer: List[SecurityEvent] = []
        self.event_handlers: List[callable] = []
        self._lock = threading.RLock()
        
        # Initialize event counters for monitoring
        self.event_counters = {
            event_type: 0 for event_type in SecurityEventType
        }
        
        # Initialize threat detection
        self.threat_patterns = self._init_threat_patterns()
        self.suspicious_activities = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for audit logging."""
        return {
            'buffer_size': 1000,
            'flush_interval': 60,  # seconds
            'log_format': 'json',
            'include_stack_trace': False,
            'encrypt_sensitive_data': True,
            'retention_days': 365,
            'compliance_mode': True,
            'real_time_alerts': True,
            'threat_detection': True
        }
        
    def _init_threat_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat detection patterns."""
        return {
            'brute_force': {
                'max_failures': 5,
                'time_window': 300,  # 5 minutes
                'risk_score': 8.0
            },
            'privilege_escalation': {
                'unauthorized_admin_access': True,
                'risk_score': 9.0
            },
            'data_exfiltration': {
                'large_data_access': 1000000,  # 1MB
                'multiple_resource_access': 20,
                'risk_score': 7.0
            },
            'suspicious_timing': {
                'after_hours_access': True,
                'weekend_access': True,
                'risk_score': 5.0
            }
        }
        
    def log_authentication_event(self, user_id: str, action: str, outcome: str,
                                ip_address: str = None, user_agent: str = None,
                                details: Dict[str, Any] = None) -> str:
        """Log authentication-related security events."""
        
        # Determine severity based on outcome and action
        if outcome == "FAILURE":
            severity = SecurityLevel.WARNING
            if action in ["login", "password_reset"]:
                # Check for brute force patterns
                self._check_brute_force(user_id, ip_address)
        else:
            severity = SecurityLevel.INFO
            
        event = self._create_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action=action,
            resource="authentication_system",
            outcome=outcome,
            details=details or {}
        )
        
        return self._log_event(event)
        
    def log_authorization_event(self, user_id: str, resource: str, action: str,
                              outcome: str, session_id: str = None,
                              ip_address: str = None, details: Dict[str, Any] = None) -> str:
        """Log authorization-related security events."""
        
        severity = SecurityLevel.WARNING if outcome == "DENIED" else SecurityLevel.INFO
        
        # Check for privilege escalation attempts
        if outcome == "DENIED" and any(keyword in action.lower() for keyword in ["admin", "root", "super"]):
            severity = SecurityLevel.CRITICAL
            
        event = self._create_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {}
        )
        
        return self._log_event(event)
        
    def log_data_access_event(self, user_id: str, resource: str, action: str,
                            data_size: int = 0, session_id: str = None,
                            ip_address: str = None, details: Dict[str, Any] = None) -> str:
        """Log data access events for compliance and monitoring."""
        
        severity = SecurityLevel.INFO
        risk_score = 0.0
        
        # Assess risk based on data size and access patterns
        if data_size > self.threat_patterns['data_exfiltration']['large_data_access']:
            severity = SecurityLevel.WARNING
            risk_score += 5.0
            
        # Check for multiple resource access pattern
        user_access_count = self._get_user_access_count(user_id, time_window=3600)
        if user_access_count > self.threat_patterns['data_exfiltration']['multiple_resource_access']:
            severity = SecurityLevel.WARNING
            risk_score += 3.0
            
        event = self._create_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            outcome="SUCCESS",
            details={**(details or {}), "data_size": data_size},
            risk_score=risk_score
        )
        
        return self._log_event(event)
        
    def log_security_violation(self, violation_type: str, description: str,
                             user_id: str = None, ip_address: str = None,
                             severity: SecurityLevel = SecurityLevel.CRITICAL,
                             details: Dict[str, Any] = None) -> str:
        """Log security violations and attacks."""
        
        event = self._create_security_event(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            action=violation_type,
            resource="security_system",
            outcome="BLOCKED",
            details={
                "description": description,
                **(details or {})
            },
            risk_score=8.0
        )
        
        # Add threat indicators
        event.threat_indicators.extend([
            violation_type,
            f"ip:{ip_address}" if ip_address else "",
            f"user:{user_id}" if user_id else ""
        ])
        
        return self._log_event(event)
        
    def log_system_change(self, user_id: str, change_type: str, resource: str,
                         before_state: Any = None, after_state: Any = None,
                         session_id: str = None, details: Dict[str, Any] = None) -> str:
        """Log system configuration or state changes."""
        
        # Hash sensitive state information
        change_details = {
            "change_type": change_type,
            **(details or {})
        }
        
        if before_state is not None:
            change_details["before_hash"] = self._hash_state(before_state)
            
        if after_state is not None:
            change_details["after_hash"] = self._hash_state(after_state)
            
        event = self._create_security_event(
            event_type=SecurityEventType.SYSTEM_CHANGE,
            severity=SecurityLevel.INFO,
            user_id=user_id,
            session_id=session_id,
            action=change_type,
            resource=resource,
            outcome="SUCCESS",
            details=change_details
        )
        
        return self._log_event(event)
        
    def log_threat_detection(self, threat_type: str, description: str,
                           risk_score: float, ip_address: str = None,
                           user_id: str = None, indicators: List[str] = None,
                           details: Dict[str, Any] = None) -> str:
        """Log threat detection alerts."""
        
        # Determine severity based on risk score
        if risk_score >= 9.0:
            severity = SecurityLevel.EMERGENCY
        elif risk_score >= 7.0:
            severity = SecurityLevel.CRITICAL
        elif risk_score >= 5.0:
            severity = SecurityLevel.WARNING
        else:
            severity = SecurityLevel.INFO
            
        event = self._create_security_event(
            event_type=SecurityEventType.THREAT_DETECTION,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            action=threat_type,
            resource="threat_detection_system",
            outcome="DETECTED",
            details={
                "description": description,
                "indicators": indicators or [],
                **(details or {})
            },
            risk_score=risk_score,
            threat_indicators=indicators or []
        )
        
        return self._log_event(event)
        
    def _create_security_event(self, **kwargs) -> SecurityEvent:
        """Create a security event with standard fields."""
        event_id = self._generate_event_id()
        timestamp = datetime.now(timezone.utc)
        
        return SecurityEvent(
            event_id=event_id,
            timestamp=timestamp,
            **kwargs
        )
        
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = str(time.time_ns())
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
        
    def _hash_state(self, state: Any) -> str:
        """Create hash of state for integrity verification."""
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()
        
    def _log_event(self, event: SecurityEvent) -> str:
        """Log security event and trigger handlers."""
        with self._lock:
            # Add to buffer
            self.event_buffer.append(event)
            
            # Update counters
            self.event_counters[event.event_type] += 1
            
            # Trigger real-time alerts for critical events
            if (self.config['real_time_alerts'] and 
                event.severity in [SecurityLevel.CRITICAL, SecurityLevel.EMERGENCY]):
                self._trigger_real_time_alert(event)
                
            # Trigger event handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
                    
            # Flush buffer if needed
            if len(self.event_buffer) >= self.config['buffer_size']:
                self._flush_events()
                
        # Log to standard logger
        self._log_to_standard_logger(event)
        
        return event.event_id
        
    def _log_to_standard_logger(self, event: SecurityEvent):
        """Log event to standard Python logger."""
        log_data = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'type': event.event_type.value,
            'severity': event.severity.value,
            'user_id': event.user_id,
            'action': event.action,
            'resource': event.resource,
            'outcome': event.outcome,
            'risk_score': event.risk_score,
            'ip_address': event.ip_address
        }
        
        log_message = f"SECURITY_EVENT: {json.dumps(log_data)}"
        
        if event.severity == SecurityLevel.EMERGENCY:
            logger.critical(log_message)
        elif event.severity == SecurityLevel.CRITICAL:
            logger.error(log_message)
        elif event.severity == SecurityLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
            
    def _trigger_real_time_alert(self, event: SecurityEvent):
        """Trigger real-time alerts for critical events."""
        alert_data = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'severity': event.severity.value,
            'type': event.event_type.value,
            'action': event.action,
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'risk_score': event.risk_score,
            'threat_indicators': event.threat_indicators
        }
        
        logger.critical(f"SECURITY_ALERT: {json.dumps(alert_data)}")
        
        # Here you could integrate with external alerting systems:
        # - Send email alerts
        # - Push to SIEM systems
        # - Notify security team via Slack/Teams
        # - Trigger automated response systems
        
    def _check_brute_force(self, user_id: str, ip_address: str = None):
        """Check for brute force attack patterns."""
        if not self.config['threat_detection']:
            return
            
        identifier = ip_address or user_id
        if not identifier:
            return
            
        current_time = time.time()
        pattern = self.threat_patterns['brute_force']
        
        # Clean old entries
        if identifier in self.suspicious_activities:
            self.suspicious_activities[identifier] = [
                timestamp for timestamp in self.suspicious_activities[identifier]
                if current_time - timestamp < pattern['time_window']
            ]
        else:
            self.suspicious_activities[identifier] = []
            
        # Add current failure
        self.suspicious_activities[identifier].append(current_time)
        
        # Check if threshold exceeded
        failure_count = len(self.suspicious_activities[identifier])
        if failure_count >= pattern['max_failures']:
            self.log_threat_detection(
                threat_type="brute_force_attack",
                description=f"Brute force attack detected: {failure_count} failed attempts",
                risk_score=pattern['risk_score'],
                ip_address=ip_address,
                user_id=user_id,
                indicators=[f"failed_attempts:{failure_count}", f"time_window:{pattern['time_window']}"]
            )
            
    def _get_user_access_count(self, user_id: str, time_window: int) -> int:
        """Get user's access count within time window."""
        current_time = time.time()
        count = 0
        
        for event in self.event_buffer:
            if (event.user_id == user_id and 
                event.event_type == SecurityEventType.DATA_ACCESS and
                (current_time - event.timestamp.timestamp()) < time_window):
                count += 1
                
        return count
        
    def _flush_events(self):
        """Flush events from buffer to persistent storage."""
        if not self.event_buffer:
            return
            
        # In a production system, you would save to:
        # - Database (PostgreSQL, MongoDB)
        # - Log files with rotation
        # - SIEM systems
        # - Cloud logging services
        
        logger.info(f"Flushing {len(self.event_buffer)} security events")
        self.event_buffer.clear()
        
    def add_event_handler(self, handler: callable):
        """Add custom event handler."""
        self.event_handlers.append(handler)
        
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        with self._lock:
            recent_events = self.event_buffer[-10:] if self.event_buffer else []
            
            return {
                'event_counters': dict(self.event_counters),
                'buffer_size': len(self.event_buffer),
                'recent_events': [
                    {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'type': event.event_type.value,
                        'severity': event.severity.value,
                        'action': event.action,
                        'outcome': event.outcome,
                        'risk_score': event.risk_score
                    }
                    for event in recent_events
                ],
                'threat_activity': {
                    identifier: len(timestamps)
                    for identifier, timestamps in self.suspicious_activities.items()
                    if timestamps
                }
            }
            
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for audit purposes."""
        relevant_events = [
            event for event in self.event_buffer
            if start_date <= event.timestamp <= end_date
        ]
        
        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(relevant_events),
            'events_by_type': {},
            'events_by_severity': {},
            'security_violations': 0,
            'authentication_failures': 0,
            'authorization_denials': 0,
            'high_risk_events': []
        }
        
        for event in relevant_events:
            # Count by type
            event_type = event.event_type.value
            report['events_by_type'][event_type] = report['events_by_type'].get(event_type, 0) + 1
            
            # Count by severity
            severity = event.severity.value
            report['events_by_severity'][severity] = report['events_by_severity'].get(severity, 0) + 1
            
            # Count specific event types
            if event.event_type == SecurityEventType.SECURITY_VIOLATION:
                report['security_violations'] += 1
                
            if (event.event_type == SecurityEventType.AUTHENTICATION and 
                event.outcome == "FAILURE"):
                report['authentication_failures'] += 1
                
            if (event.event_type == SecurityEventType.AUTHORIZATION and 
                event.outcome == "DENIED"):
                report['authorization_denials'] += 1
                
            # Collect high-risk events
            if event.risk_score >= 7.0:
                report['high_risk_events'].append({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'type': event.event_type.value,
                    'action': event.action,
                    'risk_score': event.risk_score,
                    'threat_indicators': event.threat_indicators
                })
                
        return report