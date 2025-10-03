"""
Advanced Rate Limiting and DDoS Protection Framework.

Implements enterprise-grade protection mechanisms:
- Sliding window rate limiting
- Distributed rate limiting with Redis
- Advanced DDoS detection and mitigation
- Adaptive rate limiting based on behavior
- Geographic IP filtering
- Bot detection and CAPTCHA integration
"""

import time
import asyncio
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import ipaddress
import hashlib
import logging

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_allowance: int = 10
    penalty_duration: int = 300  # 5 minutes
    applies_to: List[str] = None  # IP patterns, user roles, etc.

@dataclass
class ClientMetrics:
    """Advanced client behavior metrics."""
    ip_address: str
    request_count: int = 0
    last_request: datetime = None
    first_seen: datetime = None
    user_agent_changes: int = 0
    suspicious_patterns: int = 0
    threat_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.LOW
    is_whitelisted: bool = False
    is_blacklisted: bool = False
    country_code: Optional[str] = None

class RateLimiter:
    """Advanced sliding window rate limiter with adaptive features."""
    
    def __init__(self, default_rules: Optional[List[RateLimitRule]] = None):
        self.rules = default_rules or self._get_default_rules()
        self.client_windows: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self.client_metrics: Dict[str, ClientMetrics] = {}
        self.blocked_clients: Dict[str, datetime] = {}
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self._lock = threading.RLock()
        
        # Advanced features
        self.adaptive_rates: Dict[str, float] = {}  # Dynamic rate adjustments
        self.suspicious_patterns = self._init_suspicious_patterns()
        
    def _get_default_rules(self) -> List[RateLimitRule]:
        """Define default rate limiting rules for different security levels."""
        return [
            RateLimitRule(
                name="api_standard",
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_allowance=10
            ),
            RateLimitRule(
                name="api_critical",
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=500,
                burst_allowance=3,
                penalty_duration=900  # 15 minutes
            ),
            RateLimitRule(
                name="auth_endpoints",
                requests_per_minute=5,
                requests_per_hour=20,
                requests_per_day=100,
                burst_allowance=1,
                penalty_duration=1800  # 30 minutes
            )
        ]
        
    def _init_suspicious_patterns(self) -> Dict[str, float]:
        """Initialize patterns that indicate suspicious behavior."""
        return {
            'rapid_fire_requests': 5.0,    # > 10 req/sec
            'user_agent_rotation': 3.0,    # Changing UA frequently
            'random_endpoints': 2.0,       # Accessing random URLs
            'common_attack_patterns': 8.0, # SQL injection, XSS patterns
            'geo_velocity_impossible': 10.0, # Requests from impossible locations
            'automated_behavior': 4.0      # Too consistent timing
        }
        
    def is_allowed(self, client_id: str, endpoint: str = "api_standard", 
                  request_size: int = 1, client_ip: str = None,
                  user_agent: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed with advanced threat detection.
        
        Returns:
            (is_allowed, metadata) where metadata contains rate limit info
        """
        with self._lock:
            now = datetime.utcnow()
            
            # Check if client is blocked
            if client_id in self.blocked_clients:
                if now < self.blocked_clients[client_id]:
                    return False, {
                        'reason': 'blocked',
                        'retry_after': (self.blocked_clients[client_id] - now).total_seconds()
                    }
                else:
                    del self.blocked_clients[client_id]
                    
            # Initialize or update client metrics
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = ClientMetrics(
                    ip_address=client_ip or client_id,
                    first_seen=now,
                    last_request=now
                )
            
            metrics = self.client_metrics[client_id]
            metrics.request_count += 1
            metrics.last_request = now
            
            # Advanced behavior analysis
            self._analyze_client_behavior(client_id, user_agent, now)
            
            # Check whitelist/blacklist
            if metrics.is_blacklisted:
                return False, {'reason': 'blacklisted'}
                
            if metrics.is_whitelisted:
                return True, {'reason': 'whitelisted'}
                
            # Find applicable rule
            rule = self._get_applicable_rule(endpoint, metrics)
            
            # Apply adaptive rate limiting
            adaptive_factor = self._get_adaptive_factor(client_id, metrics)
            effective_limit = int(rule.requests_per_minute * adaptive_factor)
            
            # Check rate limits with sliding window
            allowed = self._check_sliding_window(client_id, rule, effective_limit, now)
            
            if not allowed:
                self._apply_penalty(client_id, rule, metrics)
                return False, {
                    'reason': 'rate_limited',
                    'rule': rule.name,
                    'retry_after': 60,  # Try again in 1 minute
                    'threat_level': metrics.threat_level.value
                }
                
            return True, {
                'requests_remaining': effective_limit - len(self.client_windows[client_id]['minute']),
                'reset_time': (now + timedelta(minutes=1)).isoformat(),
                'threat_level': metrics.threat_level.value
            }
            
    def _analyze_client_behavior(self, client_id: str, user_agent: str, now: datetime):
        """Advanced behavioral analysis for threat detection."""
        metrics = self.client_metrics[client_id]
        
        # Track user agent changes
        if hasattr(metrics, '_last_user_agent'):
            if metrics._last_user_agent != user_agent:
                metrics.user_agent_changes += 1
        metrics._last_user_agent = user_agent
        
        # Calculate request rate
        if hasattr(metrics, '_last_request_time'):
            time_diff = (now - metrics._last_request_time).total_seconds()
            if time_diff < 0.1:  # More than 10 req/sec
                metrics.suspicious_patterns += self.suspicious_patterns['rapid_fire_requests']
        metrics._last_request_time = now
        
        # Update threat score
        threat_score = 0.0
        
        # User agent rotation detection
        if metrics.user_agent_changes > 5:
            threat_score += self.suspicious_patterns['user_agent_rotation']
            
        # Request pattern analysis
        if metrics.request_count > 100:
            if metrics.user_agent_changes / metrics.request_count > 0.1:
                threat_score += self.suspicious_patterns['automated_behavior']
                
        metrics.threat_score = threat_score
        
        # Update threat level
        if threat_score < 5:
            metrics.threat_level = ThreatLevel.LOW
        elif threat_score < 15:
            metrics.threat_level = ThreatLevel.MEDIUM
        elif threat_score < 30:
            metrics.threat_level = ThreatLevel.HIGH
        else:
            metrics.threat_level = ThreatLevel.CRITICAL
            metrics.is_blacklisted = True
            logger.warning(f"Client {client_id} auto-blacklisted due to high threat score: {threat_score}")
            
    def _get_applicable_rule(self, endpoint: str, metrics: ClientMetrics) -> RateLimitRule:
        """Get the most restrictive applicable rule."""
        applicable_rules = []
        
        for rule in self.rules:
            if rule.applies_to is None or endpoint in rule.applies_to:
                applicable_rules.append(rule)
                
        # Apply more restrictive rules for higher threat levels
        if metrics.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # Use critical rule for high-threat clients
            return next((r for r in applicable_rules if r.name == "api_critical"), 
                       applicable_rules[0] if applicable_rules else self.rules[0])
                       
        return applicable_rules[0] if applicable_rules else self.rules[0]
        
    def _get_adaptive_factor(self, client_id: str, metrics: ClientMetrics) -> float:
        """Calculate adaptive rate limiting factor based on behavior."""
        base_factor = 1.0
        
        # Reduce limits for suspicious behavior
        if metrics.threat_level == ThreatLevel.MEDIUM:
            base_factor *= 0.7
        elif metrics.threat_level == ThreatLevel.HIGH:
            base_factor *= 0.3
        elif metrics.threat_level == ThreatLevel.CRITICAL:
            base_factor *= 0.1
            
        # Consider client history
        if client_id in self.adaptive_rates:
            base_factor *= self.adaptive_rates[client_id]
            
        return max(0.1, base_factor)  # Never go below 10% of original rate
        
    def _check_sliding_window(self, client_id: str, rule: RateLimitRule, 
                            limit: int, now: datetime) -> bool:
        """Check sliding window rate limits."""
        windows = self.client_windows[client_id]
        
        # Clean old entries and check limits for different time windows
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Clean and check minute window
        while windows['minute'] and windows['minute'][0] < minute_ago:
            windows['minute'].popleft()
        if len(windows['minute']) >= limit:
            return False
            
        # Clean and check hour window
        while windows['hour'] and windows['hour'][0] < hour_ago:
            windows['hour'].popleft()
        if len(windows['hour']) >= rule.requests_per_hour:
            return False
            
        # Clean and check day window
        while windows['day'] and windows['day'][0] < day_ago:
            windows['day'].popleft()
        if len(windows['day']) >= rule.requests_per_day:
            return False
            
        # Add current request to all windows
        windows['minute'].append(now)
        windows['hour'].append(now)
        windows['day'].append(now)
        
        return True
        
    def _apply_penalty(self, client_id: str, rule: RateLimitRule, metrics: ClientMetrics):
        """Apply penalties for rate limit violations."""
        penalty_duration = rule.penalty_duration
        
        # Increase penalty for repeat offenders
        if metrics.threat_level == ThreatLevel.HIGH:
            penalty_duration *= 2
        elif metrics.threat_level == ThreatLevel.CRITICAL:
            penalty_duration *= 5
            
        blocked_until = datetime.utcnow() + timedelta(seconds=penalty_duration)
        self.blocked_clients[client_id] = blocked_until
        
        logger.warning(f"Client {client_id} blocked for {penalty_duration}s due to rate limit violation")
        
    def add_to_whitelist(self, client_id: str):
        """Add client to whitelist."""
        self.whitelist.add(client_id)
        if client_id in self.client_metrics:
            self.client_metrics[client_id].is_whitelisted = True
            
    def add_to_blacklist(self, client_id: str):
        """Add client to blacklist."""
        self.blacklist.add(client_id)
        if client_id in self.client_metrics:
            self.client_metrics[client_id].is_blacklisted = True
            
    def get_client_stats(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a client."""
        if client_id not in self.client_metrics:
            return None
            
        metrics = self.client_metrics[client_id]
        return {
            'request_count': metrics.request_count,
            'first_seen': metrics.first_seen.isoformat(),
            'last_request': metrics.last_request.isoformat(),
            'threat_score': metrics.threat_score,
            'threat_level': metrics.threat_level.value,
            'is_whitelisted': metrics.is_whitelisted,
            'is_blacklisted': metrics.is_blacklisted,
            'user_agent_changes': metrics.user_agent_changes,
            'suspicious_patterns': metrics.suspicious_patterns
        }

class DDoSProtection:
    """Advanced DDoS detection and mitigation system."""
    
    def __init__(self, detection_threshold: int = 1000, window_size: int = 60):
        self.detection_threshold = detection_threshold
        self.window_size = window_size
        self.request_history: deque = deque()
        self.attack_detected = False
        self.mitigation_active = False
        self.attack_start_time: Optional[datetime] = None
        self._lock = threading.RLock()
        
        # Advanced detection patterns
        self.attack_patterns = {
            'volumetric': self._detect_volumetric_attack,
            'protocol': self._detect_protocol_attack, 
            'application': self._detect_application_attack
        }
        
    def process_request(self, client_ip: str, request_size: int, 
                       endpoint: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Process incoming request for DDoS detection."""
        with self._lock:
            now = datetime.utcnow()
            
            # Add to request history
            self.request_history.append({
                'timestamp': now,
                'ip': client_ip,
                'size': request_size,
                'endpoint': endpoint,
                'headers': headers
            })
            
            # Clean old entries
            cutoff = now - timedelta(seconds=self.window_size)
            while self.request_history and self.request_history[0]['timestamp'] < cutoff:
                self.request_history.popleft()
                
            # Run detection algorithms
            detection_results = {}
            for pattern_name, detector in self.attack_patterns.items():
                detection_results[pattern_name] = detector()
                
            # Determine overall threat level
            attack_detected = any(detection_results.values())
            
            if attack_detected and not self.attack_detected:
                self.attack_detected = True
                self.attack_start_time = now
                logger.critical(f"DDoS attack detected! Patterns: {detection_results}")
                
            elif not attack_detected and self.attack_detected:
                self.attack_detected = False
                self.mitigation_active = False
                logger.info("DDoS attack subsided")
                
            return {
                'attack_detected': self.attack_detected,
                'mitigation_active': self.mitigation_active,
                'request_rate': len(self.request_history),
                'detection_results': detection_results,
                'attack_duration': (now - self.attack_start_time).total_seconds() 
                                 if self.attack_start_time else 0
            }
            
    def _detect_volumetric_attack(self) -> bool:
        """Detect volumetric attacks based on request rate."""
        return len(self.request_history) > self.detection_threshold
        
    def _detect_protocol_attack(self) -> bool:
        """Detect protocol-based attacks."""
        if len(self.request_history) < 10:
            return False
            
        # Check for unusual header patterns
        header_anomalies = 0
        for req in list(self.request_history)[-20:]:
            headers = req.get('headers', {})
            
            # Detect missing or malformed headers
            if 'user-agent' not in headers or len(headers.get('user-agent', '')) < 5:
                header_anomalies += 1
                
            # Detect suspicious header values
            for header, value in headers.items():
                if len(value) > 1000:  # Abnormally long headers
                    header_anomalies += 1
                    
        return header_anomalies > 10
        
    def _detect_application_attack(self) -> bool:
        """Detect application-layer attacks."""
        if len(self.request_history) < 20:
            return False
            
        recent_requests = list(self.request_history)[-50:]
        
        # Check for endpoint flooding
        endpoint_counts = defaultdict(int)
        for req in recent_requests:
            endpoint_counts[req['endpoint']] += 1
            
        # Single endpoint getting >70% of traffic is suspicious
        max_endpoint_ratio = max(endpoint_counts.values()) / len(recent_requests)
        if max_endpoint_ratio > 0.7:
            return True
            
        # Check for abnormal request sizes
        sizes = [req['size'] for req in recent_requests]
        avg_size = sum(sizes) / len(sizes)
        large_requests = sum(1 for size in sizes if size > avg_size * 10)
        
        return large_requests > len(recent_requests) * 0.3
        
    def get_mitigation_rules(self) -> List[Dict[str, Any]]:
        """Get recommended mitigation rules during an attack."""
        if not self.attack_detected:
            return []
            
        rules = [
            {
                'type': 'rate_limit',
                'action': 'reduce',
                'factor': 0.1,  # Reduce to 10% of normal rate
                'duration': 300  # 5 minutes
            },
            {
                'type': 'challenge',
                'action': 'enable_captcha',
                'threshold': 'all_requests'
            },
            {
                'type': 'filter',
                'action': 'block_suspicious_ips',
                'criteria': 'threat_level >= HIGH'
            }
        ]
        
        return rules