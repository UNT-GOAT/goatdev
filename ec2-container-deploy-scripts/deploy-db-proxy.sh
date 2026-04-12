#!/bin/bash
set -e

prune_output=""
if ! prune_output=$(docker image prune -af --filter "until=24h" 2>&1); then
  if grep -qi "a prune operation is already running" <<< "$prune_output"; then
    echo "Skipping docker image prune: another prune operation is already running"
  else
    echo "$prune_output"
    exit 1
  fi
elif [ -n "$prune_output" ]; then
  echo "$prune_output"
fi

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 937249941844.dkr.ecr.us-east-2.amazonaws.com


IMAGE=937249941844.dkr.ecr.us-east-2.amazonaws.com/herdsync-db-proxy:latest

docker pull $IMAGE
docker rm -f db-proxy || true

docker run -d \
  --name db-proxy \
  --restart unless-stopped \
  --network herdsync-net \
  -p 8003:8003 \
  -v auth-keys:/app/keys \
  --env-file /home/ubuntu/db-proxy.env \
  $IMAGE

echo "Waiting for db-proxy service to start..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8003/health > /dev/null 2>&1; then
    echo "db-proxy service is healthy"
    exit 0
  fi
  echo "  attempt $i/30 — not ready yet"
  sleep 2
done
echo "db-proxy service failed to start within 60s"
docker logs db-proxy --tail 30
exit 1
