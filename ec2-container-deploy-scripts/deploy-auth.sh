#!/bin/bash
set -e
docker image prune -af --filter "until=24h"

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 937249941844.dkr.ecr.us-east-2.amazonaws.com


IMAGE=937249941844.dkr.ecr.us-east-2.amazonaws.com/herdsync-auth:latest

docker pull $IMAGE
docker rm -f herdsync-auth || true

docker run -d \
  --name herdsync-auth \
  --restart unless-stopped \
  --network herdsync-net \
  -p 8001:8001 \
  -v auth-keys:/app/keys \
  --env-file /home/ubuntu/auth.env \
  $IMAGE

echo "Waiting for auth service to start..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8001/auth/health > /dev/null 2>&1; then
    echo "Auth service is healthy"
    exit 0
  fi
  echo "  attempt $i/30 — not ready yet"
  sleep 2
done
echo "Auth service failed to start within 60s"
docker logs herdsync-auth --tail 30
exit 1