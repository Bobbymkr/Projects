"""
Export OpenAPI Schema Script.

Exports the OpenAPI 3.0 schema from the FastAPI application
to a JSON file for use in API documentation and client generation.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.api.main import app
    
    def export_openapi_schema(output_path: str = "docs/api/openapi.json"):
        """
        Export OpenAPI schema to JSON file.
        
        Args:
            output_path: Path to output file
        """
        try:
            # Get OpenAPI schema from FastAPI app
            openapi_schema = app.openapi()
            
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write schema to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
            
            print(f"✓ OpenAPI schema exported to {output_path}")
            print(f"  - Version: {openapi_schema.get('info', {}).get('version', 'unknown')}")
            print(f"  - Endpoints: {len(openapi_schema.get('paths', {}))}")
            
            return True
        except Exception as e:
            print(f"✗ Error exporting OpenAPI schema: {e}")
            return False
    
    if __name__ == "__main__":
        import argparse
        
        parser = argparse.ArgumentParser(description="Export OpenAPI schema")
        parser.add_argument(
            "--output",
            default="docs/api/openapi.json",
            help="Output file path"
        )
        
        args = parser.parse_args()
        success = export_openapi_schema(args.output)
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"Warning: Could not import FastAPI app: {e}")
    print("OpenAPI schema export requires FastAPI to be installed.")
    sys.exit(1)

