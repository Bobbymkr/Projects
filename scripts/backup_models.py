#!/usr/bin/env python3
"""
Week 8: Automated Model Backup Script.

Backs up trained models to S3/GCS.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def backup_to_s3(models_dir: Path, bucket_name: str, prefix: str = "models"):
    """Backup models to S3."""
    if not AWS_AVAILABLE:
        raise ImportError("boto3 not available. Install with: pip install boto3")
    
    s3_client = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt")) + list(models_dir.glob("*.h5"))
    
    for model_file in model_files:
        s3_key = f"{prefix}/{timestamp}/{model_file.name}"
        s3_client.upload_file(str(model_file), bucket_name, s3_key)
        print(f"Uploaded {model_file.name} to s3://{bucket_name}/{s3_key}")


def backup_to_gcs(models_dir: Path, bucket_name: str, prefix: str = "models"):
    """Backup models to GCS."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage not available. Install with: pip install google-cloud-storage")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt")) + list(models_dir.glob("*.h5"))
    
    for model_file in model_files:
        blob_name = f"{prefix}/{timestamp}/{model_file.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(model_file))
        print(f"Uploaded {model_file.name} to gs://{bucket_name}/{blob_name}")


def main():
    parser = argparse.ArgumentParser(description="Backup models to cloud storage")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Models directory")
    parser.add_argument("--provider", choices=["s3", "gcs"], required=True, help="Cloud provider")
    parser.add_argument("--bucket", required=True, help="Bucket name")
    parser.add_argument("--prefix", default="models", help="Prefix for backup")
    
    args = parser.parse_args()
    
    if args.provider == "s3":
        backup_to_s3(args.models_dir, args.bucket, args.prefix)
    elif args.provider == "gcs":
        backup_to_gcs(args.models_dir, args.bucket, args.prefix)


if __name__ == "__main__":
    main()

