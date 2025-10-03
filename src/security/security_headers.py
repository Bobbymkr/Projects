"""
Advanced Security Headers Implementation.

Implements comprehensive HTTP security headers for web application protection:
- Content Security Policy (CSP)
- HTTP Strict Transport Security (HSTS)
- X-Frame-Options protection
- X-Content-Type-Options
- Referrer Policy controls
- Feature Policy restrictions
- And more security headers
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CSPDirective(Enum):
    """Content Security Policy directive types."""
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    CONNECT_SRC = "connect-src"
    FONT_SRC = "font-src"
    OBJECT_SRC = "object-src"
    MEDIA_SRC = "media-src"
    FRAME_SRC = "frame-src"
    CHILD_SRC = "child-src"
    FORM_ACTION = "form-action"
    FRAME_ANCESTORS = "frame-ancestors"
    BASE_URI = "base-uri"

@dataclass
class SecurityHeaderConfig:
    """Configuration for security headers."""
    enable_hsts: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True
    
    enable_csp: bool = True
    csp_report_only: bool = False
    csp_report_uri: Optional[str] = None
    
    enable_frame_options: bool = True
    frame_options_value: str = "DENY"
    
    enable_content_type_options: bool = True
    enable_xss_protection: bool = True
    enable_referrer_policy: bool = True
    referrer_policy_value: str = "strict-origin-when-cross-origin"
    
    enable_feature_policy: bool = True
    enable_permissions_policy: bool = True

class SecurityHeaders:
    """Advanced security headers manager."""
    
    def __init__(self, config: Optional[SecurityHeaderConfig] = None):
        self.config = config or SecurityHeaderConfig()
        self.csp_policy: Dict[str, List[str]] = {}
        self.feature_policy: Dict[str, str] = {}
        self.permissions_policy: Dict[str, List[str]] = {}
        
        # Initialize default policies
        self._init_default_csp()
        self._init_default_feature_policy()
        self._init_default_permissions_policy()
        
    def _init_default_csp(self):
        """Initialize secure default Content Security Policy."""
        self.csp_policy = {
            CSPDirective.DEFAULT_SRC.value: ["'self'"],
            CSPDirective.SCRIPT_SRC.value: [
                "'self'",
                "'unsafe-inline'",  # Only for development - remove in production
                "https://cdnjs.cloudflare.com",
                "https://cdn.jsdelivr.net"
            ],
            CSPDirective.STYLE_SRC.value: [
                "'self'",
                "'unsafe-inline'",  # Often needed for frameworks
                "https://fonts.googleapis.com",
                "https://cdnjs.cloudflare.com"
            ],
            CSPDirective.IMG_SRC.value: [
                "'self'",
                "data:",
                "https:"
            ],
            CSPDirective.CONNECT_SRC.value: [
                "'self'",
                "https:",
                "wss:"
            ],
            CSPDirective.FONT_SRC.value: [
                "'self'",
                "data:",
                "https://fonts.gstatic.com"
            ],
            CSPDirective.OBJECT_SRC.value: ["'none'"],
            CSPDirective.FRAME_SRC.value: ["'none'"],
            CSPDirective.FRAME_ANCESTORS.value: ["'none'"],
            CSPDirective.BASE_URI.value: ["'self'"],
            CSPDirective.FORM_ACTION.value: ["'self'"]
        }
        
    def _init_default_feature_policy(self):
        """Initialize secure default Feature Policy."""
        self.feature_policy = {
            "camera": "'none'",
            "microphone": "'none'",
            "geolocation": "'none'",
            "payment": "'none'",
            "usb": "'none'",
            "accelerometer": "'none'",
            "gyroscope": "'none'",
            "magnetometer": "'none'",
            "fullscreen": "'self'",
            "autoplay": "'none'"
        }
        
    def _init_default_permissions_policy(self):
        """Initialize secure default Permissions Policy."""
        self.permissions_policy = {
            "camera": [],
            "microphone": [],
            "geolocation": [],
            "payment": [],
            "usb": [],
            "accelerometer": [],
            "gyroscope": [],
            "magnetometer": [],
            "fullscreen": ["self"],
            "autoplay": [],
            "encrypted-media": [],
            "picture-in-picture": []
        }
        
    def add_csp_directive(self, directive: str, sources: List[str]):
        """Add or update a CSP directive."""
        if directive not in self.csp_policy:
            self.csp_policy[directive] = []
        
        for source in sources:
            if source not in self.csp_policy[directive]:
                self.csp_policy[directive].append(source)
                
        logger.info(f"Added CSP directive {directive}: {sources}")
        
    def remove_csp_directive(self, directive: str):
        """Remove a CSP directive."""
        if directive in self.csp_policy:
            del self.csp_policy[directive]
            logger.info(f"Removed CSP directive: {directive}")
            
    def set_csp_nonce(self, directive: str, nonce: str):
        """Add nonce to CSP directive for inline scripts/styles."""
        nonce_value = f"'nonce-{nonce}'"
        
        if directive not in self.csp_policy:
            self.csp_policy[directive] = []
            
        # Remove any existing nonce values
        self.csp_policy[directive] = [
            src for src in self.csp_policy[directive] 
            if not src.startswith("'nonce-")
        ]
        
        self.csp_policy[directive].append(nonce_value)
        
    def build_csp_header(self) -> str:
        """Build the Content Security Policy header value."""
        directives = []
        
        for directive, sources in self.csp_policy.items():
            if sources:
                directive_value = f"{directive} {' '.join(sources)}"
                directives.append(directive_value)
                
        csp_value = "; ".join(directives)
        
        # Add report URI if configured
        if self.config.csp_report_uri:
            csp_value += f"; report-uri {self.config.csp_report_uri}"
            
        return csp_value
        
    def build_feature_policy_header(self) -> str:
        """Build the Feature Policy header value."""
        policies = []
        
        for feature, allowlist in self.feature_policy.items():
            if allowlist == "'none'":
                policies.append(f"{feature} 'none'")
            elif allowlist == "'self'":
                policies.append(f"{feature} 'self'")
            else:
                policies.append(f"{feature} {allowlist}")
                
        return ", ".join(policies)
        
    def build_permissions_policy_header(self) -> str:
        """Build the Permissions Policy header value."""
        policies = []
        
        for directive, allowlist in self.permissions_policy.items():
            if not allowlist:
                policies.append(f"{directive}=()")
            elif allowlist == ["self"]:
                policies.append(f"{directive}=(self)")
            else:
                formatted_list = " ".join(f'"{item}"' for item in allowlist)
                policies.append(f"{directive}=({formatted_list})")
                
        return ", ".join(policies)
        
    def get_all_headers(self, request_is_https: bool = False) -> Dict[str, str]:
        """Get all security headers as a dictionary."""
        headers = {}
        
        # HTTP Strict Transport Security
        if self.config.enable_hsts and request_is_https:
            hsts_value = f"max-age={self.config.hsts_max_age}"
            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.config.hsts_preload:
                hsts_value += "; preload"
            headers["Strict-Transport-Security"] = hsts_value
            
        # Content Security Policy
        if self.config.enable_csp:
            csp_header = "Content-Security-Policy-Report-Only" if self.config.csp_report_only else "Content-Security-Policy"
            headers[csp_header] = self.build_csp_header()
            
        # X-Frame-Options
        if self.config.enable_frame_options:
            headers["X-Frame-Options"] = self.config.frame_options_value
            
        # X-Content-Type-Options
        if self.config.enable_content_type_options:
            headers["X-Content-Type-Options"] = "nosniff"
            
        # X-XSS-Protection (legacy but still useful)
        if self.config.enable_xss_protection:
            headers["X-XSS-Protection"] = "1; mode=block"
            
        # Referrer Policy
        if self.config.enable_referrer_policy:
            headers["Referrer-Policy"] = self.config.referrer_policy_value
            
        # Feature Policy (legacy)
        if self.config.enable_feature_policy:
            headers["Feature-Policy"] = self.build_feature_policy_header()
            
        # Permissions Policy (modern replacement for Feature Policy)
        if self.config.enable_permissions_policy:
            headers["Permissions-Policy"] = self.build_permissions_policy_header()
            
        # Additional security headers
        headers.update({
            "X-Permitted-Cross-Domain-Policies": "none",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin", 
            "Cross-Origin-Resource-Policy": "same-origin",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        })
        
        return headers
        
    def validate_headers(self) -> List[str]:
        """Validate current security header configuration."""
        warnings = []
        
        # Check for potential security issues
        if "'unsafe-inline'" in self.csp_policy.get(CSPDirective.SCRIPT_SRC.value, []):
            warnings.append("CSP allows unsafe-inline scripts - consider using nonces or hashes")
            
        if "'unsafe-eval'" in self.csp_policy.get(CSPDirective.SCRIPT_SRC.value, []):
            warnings.append("CSP allows unsafe-eval - this is a security risk")
            
        if not self.config.enable_hsts:
            warnings.append("HSTS is disabled - consider enabling for HTTPS sites")
            
        if self.config.frame_options_value not in ["DENY", "SAMEORIGIN"]:
            warnings.append("X-Frame-Options should be DENY or SAMEORIGIN")
            
        if not self.config.enable_csp:
            warnings.append("Content Security Policy is disabled")
            
        return warnings
        
    def create_csp_report_handler(self):
        """Create a handler for CSP violation reports."""
        def handle_csp_report(report_data: Dict[str, Any]):
            """Handle CSP violation report."""
            try:
                violation = report_data.get("csp-report", {})
                
                logger.warning(
                    f"CSP Violation: "
                    f"blocked-uri={violation.get('blocked-uri', 'unknown')}, "
                    f"violated-directive={violation.get('violated-directive', 'unknown')}, "
                    f"document-uri={violation.get('document-uri', 'unknown')}, "
                    f"source-file={violation.get('source-file', 'unknown')}"
                )
                
                # You can implement additional logic here like:
                # - Store violations in database
                # - Send alerts for critical violations
                # - Auto-adjust CSP policy based on legitimate violations
                
            except Exception as e:
                logger.error(f"Error processing CSP report: {e}")
                
        return handle_csp_report
        
    def get_production_config(self) -> 'SecurityHeaderConfig':
        """Get production-ready security header configuration."""
        return SecurityHeaderConfig(
            enable_hsts=True,
            hsts_max_age=31536000,  # 1 year
            hsts_include_subdomains=True,
            hsts_preload=True,
            
            enable_csp=True,
            csp_report_only=False,  # Enforce CSP in production
            
            enable_frame_options=True,
            frame_options_value="DENY",
            
            enable_content_type_options=True,
            enable_xss_protection=True,
            enable_referrer_policy=True,
            referrer_policy_value="strict-origin-when-cross-origin",
            
            enable_feature_policy=True,
            enable_permissions_policy=True
        )
        
    def get_development_config(self) -> 'SecurityHeaderConfig':
        """Get development-friendly security header configuration."""
        return SecurityHeaderConfig(
            enable_hsts=False,  # Don't enforce HSTS in development
            
            enable_csp=True,
            csp_report_only=True,  # Report-only mode for development
            
            enable_frame_options=True,
            frame_options_value="SAMEORIGIN",  # Allow framing from same origin
            
            enable_content_type_options=True,
            enable_xss_protection=True,
            enable_referrer_policy=True,
            referrer_policy_value="unsafe-url",  # More permissive for development
            
            enable_feature_policy=False,  # Disable for easier development
            enable_permissions_policy=False
        )