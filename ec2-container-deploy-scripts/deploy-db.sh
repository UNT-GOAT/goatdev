#!/bin/bash
set -e
docker image prune -af --filter "until=24h"

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 937249941844.dkr.ecr.us-east-2.amazonaws.com


IMAGE=937249941844.dkr.ecr.us-east-2.amazonaws.com/herdsync-db:latest

docker pull $IMAGE
docker rm -f db || true

docker run -d \
  --name db \
  --restart unless-stopped \
  --network herdsync-net \
  -p 8002:8002 \
  --env-file /home/ubuntu/db.env \
  $IMAGE

echo "Waiting for db service to start..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8002/health > /dev/null 2>&1; then
    echo "db service is healthy"
    exit 0
  fi
  echo "  attempt $i/30 — not ready yet"
  sleep 2
done
echo "db service failed to start within 60s"
docker logs db --tail 30
exit 1
