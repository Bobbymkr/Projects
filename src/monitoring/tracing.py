"""
Week 5: Distributed Tracing with Jaeger/OpenTelemetry.

Adds tracing to:
- Camera → Detection pipeline
- Agent decision flow
- Signal control execution
"""

import functools
from typing import Optional, Any, Callable

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Create dummy tracer
    class DummyTracer:
        def start_as_current_span(self, *args, **kwargs):
            return DummySpan()
    class DummySpan:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def set_attribute(self, *args, **kwargs): pass
        def set_status(self, *args, **kwargs): pass
    trace = type('obj', (object,), {'get_tracer': lambda *args, **kwargs: DummyTracer()})()


def setup_tracing(service_name: str = "adaptive-traffic", jaeger_endpoint: Optional[str] = None):
    """Setup OpenTelemetry tracing."""
    if not OPENTELEMETRY_AVAILABLE:
        print("Warning: OpenTelemetry not available. Tracing disabled.")
        return
    
    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0"
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add exporter
    if jaeger_endpoint:
        exporter = JaegerExporter(
            agent_host_name=jaeger_endpoint.split(':')[0],
            agent_port=int(jaeger_endpoint.split(':')[1]) if ':' in jaeger_endpoint else 6831
        )
    else:
        # Use OTLP exporter (for Jaeger with OTLP)
        exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
    
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    print(f"Tracing initialized for service: {service_name}")


def get_tracer(name: str = "adaptive-traffic"):
    """Get tracer instance."""
    return trace.get_tracer(name)


def trace_function(span_name: Optional[str] = None):
    """Decorator to trace function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name_actual = span_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name_actual) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


class TraceContext:
    """Context manager for tracing."""
    
    def __init__(self, span_name: str, attributes: Optional[dict] = None):
        self.span_name = span_name
        self.attributes = attributes or {}
        self.tracer = get_tracer()
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_as_current_span(self.span_name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, str(value))
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            self.span.record_exception(exc_val)
        else:
            self.span.set_status(trace.Status(trace.StatusCode.OK))
        return False


# Initialize tracing on import
if OPENTELEMETRY_AVAILABLE:
    setup_tracing()

