"""
Advanced Input Validation and Sanitization Framework.

Implements enterprise-grade input validation:
- SQL injection prevention
- XSS protection and HTML sanitization
- Path traversal prevention
- Command injection protection
- Schema validation
- Data type validation
- Content filtering
"""

import re
import html
import json
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Pattern, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class SanitizationLevel(Enum):
    """Sanitization security levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"

@dataclass
class ValidationRule:
    """Input validation rule configuration."""
    name: str
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    forbidden_chars: Optional[str] = None
    custom_validator: Optional[Callable] = None
    sanitize: bool = True
    required: bool = True

class SecureInputValidator:
    """Advanced input validation and sanitization system."""
    
    def __init__(self, sanitization_level: SanitizationLevel = SanitizationLevel.STRICT):
        self.sanitization_level = sanitization_level
        self.validation_rules: Dict[str, ValidationRule] = {}
        
        # Initialize security patterns
        self.security_patterns = self._init_security_patterns()
        self.html_tags_allowed = self._get_allowed_html_tags()
        
        # Initialize validation rules
        self._init_default_rules()
        
    def _init_security_patterns(self) -> Dict[str, Pattern]:
        """Initialize regex patterns for security detection."""
        patterns = {
            # SQL Injection patterns
            'sql_injection': re.compile(
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)|"
                r"(--|;|\/\*|\*\/)|"
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)|"
                r"(\'\s*(OR|AND)\s+\'\d+\'\s*=\s*\'\d+\')",
                re.IGNORECASE
            ),
            
            # XSS patterns
            'xss_script': re.compile(
                r"<\s*script[^>]*>.*?<\s*/\s*script\s*>|"
                r"javascript\s*:|"
                r"on\w+\s*=|"
                r"<\s*iframe[^>]*>|"
                r"<\s*object[^>]*>|"
                r"<\s*embed[^>]*>",
                re.IGNORECASE | re.DOTALL
            ),
            
            # Path traversal patterns
            'path_traversal': re.compile(
                r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c|"
                r"\.\.%2f|\.\.%5c|%2e%2e/|%2e%2e\\)|"
                r"(/etc/passwd|/etc/shadow|/etc/hosts|"
                r"c:\\windows\\system32|c:/windows/system32)",
                re.IGNORECASE
            ),
            
            # Command injection patterns
            'command_injection': re.compile(
                r"[;&|`$(){}[\]<>]|"
                r"\b(cat|ls|dir|type|more|head|tail|grep|find|curl|wget|nc|netcat|telnet|ssh)\b",
                re.IGNORECASE
            ),
            
            # LDAP injection patterns
            'ldap_injection': re.compile(
                r"[()&|!*]|"
                r"\\[0-9a-f]{2}",
                re.IGNORECASE
            ),
            
            # NoSQL injection patterns
            'nosql_injection': re.compile(
                r"\$where|\$ne|\$gt|\$lt|\$gte|\$lte|\$in|\$nin|\$exists|\$regex|"
                r"{\s*\$.*?}",
                re.IGNORECASE
            )
        }
        
        return patterns
        
    def _get_allowed_html_tags(self) -> Dict[SanitizationLevel, List[str]]:
        """Get allowed HTML tags for different sanitization levels."""
        return {
            SanitizationLevel.BASIC: [
                'p', 'br', 'strong', 'em', 'u', 'b', 'i', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'ul', 'ol', 'li', 'a', 'img', 'div', 'span', 'table', 'tr', 'td', 'th'
            ],
            SanitizationLevel.STRICT: [
                'p', 'br', 'strong', 'em', 'b', 'i', 'ul', 'ol', 'li'
            ],
            SanitizationLevel.PARANOID: []
        }
        
    def _init_default_rules(self):
        """Initialize default validation rules."""
        self.validation_rules = {
            'email': ValidationRule(
                name='email',
                pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                max_length=254
            ),
            'username': ValidationRule(
                name='username',
                pattern=r'^[a-zA-Z0-9_-]{3,50}$',
                min_length=3,
                max_length=50,
                allowed_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
            ),
            'password': ValidationRule(
                name='password',
                min_length=12,
                max_length=128,
                custom_validator=self._validate_password_strength
            ),
            'api_key': ValidationRule(
                name='api_key',
                pattern=r'^[a-zA-Z0-9_-]{32,}$',
                min_length=32,
                max_length=128
            ),
            'filename': ValidationRule(
                name='filename',
                pattern=r'^[a-zA-Z0-9._-]{1,255}$',
                max_length=255,
                forbidden_chars='<>:"/\\|?*'
            ),
            'ip_address': ValidationRule(
                name='ip_address',
                custom_validator=self._validate_ip_address
            ),
            'url': ValidationRule(
                name='url',
                custom_validator=self._validate_url,
                max_length=2048
            )
        }
        
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.validation_rules[rule.name] = rule
        logger.info(f"Added validation rule: {rule.name}")
        
    def validate_input(self, value: Any, rule_name: str, 
                      field_name: Optional[str] = None) -> Any:
        """
        Validate input against specified rule.
        
        Args:
            value: Input value to validate
            rule_name: Name of validation rule to apply
            field_name: Optional field name for error messages
            
        Returns:
            Validated and sanitized value
            
        Raises:
            ValidationError: If validation fails
        """
        if rule_name not in self.validation_rules:
            raise ValidationError(f"Unknown validation rule: {rule_name}")
            
        rule = self.validation_rules[rule_name]
        field_name = field_name or rule_name
        
        # Handle None/empty values
        if value is None or value == "":
            if rule.required:
                raise ValidationError(f"{field_name} is required")
            return value
            
        # Convert to string for validation
        str_value = str(value)
        
        # Check for security threats
        self._check_security_threats(str_value, field_name)
        
        # Apply length constraints
        if rule.min_length and len(str_value) < rule.min_length:
            raise ValidationError(f"{field_name} must be at least {rule.min_length} characters")
            
        if rule.max_length and len(str_value) > rule.max_length:
            raise ValidationError(f"{field_name} must not exceed {rule.max_length} characters")
            
        # Check allowed/forbidden characters
        if rule.allowed_chars:
            invalid_chars = set(str_value) - set(rule.allowed_chars)
            if invalid_chars:
                raise ValidationError(f"{field_name} contains invalid characters: {invalid_chars}")
                
        if rule.forbidden_chars:
            found_forbidden = set(str_value) & set(rule.forbidden_chars)
            if found_forbidden:
                raise ValidationError(f"{field_name} contains forbidden characters: {found_forbidden}")
                
        # Apply regex pattern
        if rule.pattern:
            if not re.match(rule.pattern, str_value):
                raise ValidationError(f"{field_name} format is invalid")
                
        # Apply custom validator
        if rule.custom_validator:
            try:
                rule.custom_validator(str_value)
            except Exception as e:
                raise ValidationError(f"{field_name} validation failed: {str(e)}")
                
        # Sanitize if required
        if rule.sanitize:
            str_value = self.sanitize_input(str_value)
            
        return str_value
        
    def _check_security_threats(self, value: str, field_name: str):
        """Check input for common security threats."""
        threats_found = []
        
        for threat_type, pattern in self.security_patterns.items():
            if pattern.search(value):
                threats_found.append(threat_type)
                
        if threats_found:
            logger.warning(f"Security threats detected in {field_name}: {threats_found}")
            
            if self.sanitization_level == SanitizationLevel.PARANOID:
                raise ValidationError(f"{field_name} contains potential security threats: {threats_found}")
                
    def sanitize_input(self, value: str) -> str:
        """Sanitize input based on sanitization level."""
        if not isinstance(value, str):
            value = str(value)
            
        # HTML escape
        value = html.escape(value)
        
        # URL encode special characters if needed
        if self.sanitization_level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            value = urllib.parse.quote(value, safe='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.~')
            
        return value
        
    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content by removing dangerous elements."""
        if not isinstance(html_content, str):
            return ""
            
        allowed_tags = self.html_tags_allowed.get(self.sanitization_level, [])
        
        if not allowed_tags:
            # Strip all HTML tags
            return re.sub(r'<[^>]+>', '', html_content)
            
        # Basic HTML sanitization (simplified - in production use a library like bleach)
        # Remove script tags and their content
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous attributes
        dangerous_attrs = ['onload', 'onclick', 'onerror', 'onmouseover', 'onmouseout', 'onfocus', 'onblur']
        for attr in dangerous_attrs:
            html_content = re.sub(f'{attr}=["\'][^"\']*["\']', '', html_content, flags=re.IGNORECASE)
            
        # Remove javascript: protocols
        html_content = re.sub(r'javascript:', '', html_content, flags=re.IGNORECASE)
        
        return html_content
        
    def _validate_password_strength(self, password: str):
        """Validate password strength."""
        errors = []
        
        if len(password) < 12:
            errors.append("Password must be at least 12 characters long")
            
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain uppercase letters")
            
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain lowercase letters")
            
        if not re.search(r'\d', password):
            errors.append("Password must contain digits")
            
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
            errors.append("Password must contain special characters")
            
        # Check for common patterns
        if re.search(r'(.)\1{2,}', password):
            errors.append("Password cannot contain repeated characters")
            
        if re.search(r'(012|123|234|345|456|567|678|789|890|abc|bcd|cde|def)', password.lower()):
            errors.append("Password cannot contain sequential characters")
            
        common_passwords = [
            'password', '123456', 'qwerty', 'admin', 'letmein', 'welcome',
            'monkey', 'dragon', 'password123', 'admin123'
        ]
        
        if password.lower() in common_passwords:
            errors.append("Password is too common")
            
        if errors:
            raise ValidationError("; ".join(errors))
            
    def _validate_ip_address(self, ip: str):
        """Validate IP address format."""
        # IPv4 validation
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ipv4_pattern, ip):
            parts = ip.split('.')
            for part in parts:
                if not (0 <= int(part) <= 255):
                    raise ValidationError("Invalid IPv4 address")
            return
            
        # IPv6 validation (simplified)
        ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        if re.match(ipv6_pattern, ip):
            return
            
        raise ValidationError("Invalid IP address format")
        
    def _validate_url(self, url: str):
        """Validate URL format and security."""
        # Basic URL pattern
        url_pattern = r'^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?$'
        if not re.match(url_pattern, url):
            raise ValidationError("Invalid URL format")
            
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        url_lower = url.lower()
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                raise ValidationError(f"Dangerous protocol detected: {protocol}")
                
        # Check for localhost/private IP access
        if self.sanitization_level == SanitizationLevel.PARANOID:
            if any(host in url_lower for host in ['localhost', '127.0.0.1', '0.0.0.0', '::1']):
                raise ValidationError("Access to localhost is not allowed")
                
    def validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> Any:
        """Validate data against JSON schema."""
        def validate_field(value, field_schema, field_path=""):
            field_type = field_schema.get('type')
            
            # Type validation
            if field_type == 'string' and not isinstance(value, str):
                raise ValidationError(f"Field {field_path} must be a string")
            elif field_type == 'integer' and not isinstance(value, int):
                raise ValidationError(f"Field {field_path} must be an integer")
            elif field_type == 'number' and not isinstance(value, (int, float)):
                raise ValidationError(f"Field {field_path} must be a number")
            elif field_type == 'boolean' and not isinstance(value, bool):
                raise ValidationError(f"Field {field_path} must be a boolean")
            elif field_type == 'array' and not isinstance(value, list):
                raise ValidationError(f"Field {field_path} must be an array")
            elif field_type == 'object' and not isinstance(value, dict):
                raise ValidationError(f"Field {field_path} must be an object")
                
            # String constraints
            if field_type == 'string':
                min_length = field_schema.get('minLength')
                max_length = field_schema.get('maxLength')
                
                if min_length and len(value) < min_length:
                    raise ValidationError(f"Field {field_path} must be at least {min_length} characters")
                if max_length and len(value) > max_length:
                    raise ValidationError(f"Field {field_path} must not exceed {max_length} characters")
                    
                pattern = field_schema.get('pattern')
                if pattern and not re.match(pattern, value):
                    raise ValidationError(f"Field {field_path} format is invalid")
                    
            # Number constraints
            if field_type in ['integer', 'number']:
                minimum = field_schema.get('minimum')
                maximum = field_schema.get('maximum')
                
                if minimum is not None and value < minimum:
                    raise ValidationError(f"Field {field_path} must be at least {minimum}")
                if maximum is not None and value > maximum:
                    raise ValidationError(f"Field {field_path} must not exceed {maximum}")
                    
            # Array constraints
            if field_type == 'array':
                min_items = field_schema.get('minItems')
                max_items = field_schema.get('maxItems')
                
                if min_items and len(value) < min_items:
                    raise ValidationError(f"Field {field_path} must have at least {min_items} items")
                if max_items and len(value) > max_items:
                    raise ValidationError(f"Field {field_path} must not have more than {max_items} items")
                    
                # Validate array items
                items_schema = field_schema.get('items')
                if items_schema:
                    for i, item in enumerate(value):
                        validate_field(item, items_schema, f"{field_path}[{i}]")
                        
            # Object validation
            if field_type == 'object':
                properties = field_schema.get('properties', {})
                required = field_schema.get('required', [])
                
                # Check required fields
                for req_field in required:
                    if req_field not in value:
                        raise ValidationError(f"Required field {field_path}.{req_field} is missing")
                        
                # Validate properties
                for prop_name, prop_value in value.items():
                    if prop_name in properties:
                        validate_field(prop_value, properties[prop_name], f"{field_path}.{prop_name}")
                        
        try:
            validate_field(data, schema)
            return data
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Schema validation error: {str(e)}")
            
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of current validation configuration."""
        return {
            'sanitization_level': self.sanitization_level.value,
            'rules_count': len(self.validation_rules),
            'available_rules': list(self.validation_rules.keys()),
            'security_patterns': list(self.security_patterns.keys()),
            'allowed_html_tags': self.html_tags_allowed.get(self.sanitization_level, [])
        }