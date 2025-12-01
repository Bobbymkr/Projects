"""
GraphQL API Route Handler.

Provides GraphQL endpoint with query, mutation, and subscription support.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
import logging

from ..dependencies import rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()

# Try to import GraphQL router, but handle gracefully if not available
try:
    from strawberry.fastapi import GraphQLRouter
    from ..graphql.schema import schema
    
    # Create GraphQL router
    graphql_router = GraphQLRouter(schema, graphiql=True)
    
    # Include GraphQL router (handles POST requests to /graphql)
    router.add_api_route(
        "/graphql",
        graphql_router,
        methods=["GET", "POST"],
        dependencies=[Depends(rate_limit)],
    )
    
    GRAPHQL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GraphQL not available: {e}. Install strawberry-graphql to enable.")
    GRAPHQL_AVAILABLE = False


@router.get("/graphql")
async def graphql_playground():
    """GraphQL Playground interface."""
    if not GRAPHQL_AVAILABLE:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GraphQL Not Available</title>
            </head>
            <body>
                <h1>GraphQL Not Available</h1>
                <p>GraphQL support requires strawberry-graphql package.</p>
                <p>Install with: <code>pip install strawberry-graphql[fastapi]</code></p>
            </body>
            </html>
            """,
            status_code=503,
        )
    
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphQL Playground</title>
    </head>
    <body>
        <h1>GraphQL Playground</h1>
        <p>GraphQL endpoint available at <code>/api/v1/graphql</code></p>
        <p>Use GraphiQL interface for interactive queries.</p>
    </body>
    </html>
    """)
