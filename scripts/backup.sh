#!/bin/bash
# Backup Script for Adaptive Traffic Control System
# Phase 4: Infrastructure Excellence

set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/backup}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RETENTION_DAYS="${RETENTION_DAYS:-7}"

echo "Starting backup process at $(date)"

# PostgreSQL Backup
if [ "${ENABLE_POSTGRES_BACKUP:-true}" == "true" ]; then
    echo "Backing up PostgreSQL database..."
    pg_dump "${DATABASE_URL}" \
        --format=custom \
        --file="${BACKUP_DIR}/postgres-${TIMESTAMP}.dump"
    
    echo "PostgreSQL backup completed: postgres-${TIMESTAMP}.dump"
fi

# Redis Backup
if [ "${ENABLE_REDIS_BACKUP:-true}" == "true" ]; then
    echo "Backing up Redis..."
    redis-cli -h "${REDIS_HOST:-localhost}" \
        -p "${REDIS_PORT:-6379}" \
        ${REDIS_PASSWORD:+-a "$REDIS_PASSWORD"} \
        --rdb "${BACKUP_DIR}/redis-${TIMESTAMP}.rdb"
    
    echo "Redis backup completed: redis-${TIMESTAMP}.rdb"
fi

# Cleanup old backups
echo "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "*.dump" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}" -name "*.rdb" -mtime +${RETENTION_DAYS} -delete

echo "Backup process completed at $(date)"

