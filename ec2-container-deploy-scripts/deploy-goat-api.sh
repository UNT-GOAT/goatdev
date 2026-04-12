#!/bin/bash
set -e

LOCK_FILE="/tmp/herdsync-ec2-deploy.lock"
exec 9>"$LOCK_FILE"
echo "Waiting for EC2 deploy lock..."
flock 9
echo "Acquired EC2 deploy lock"

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


IMAGE=937249941844.dkr.ecr.us-east-2.amazonaws.com/goat-api:latest

docker pull $IMAGE
docker rm -f goat-api || true

docker run -d \
  --name goat-api \
  --restart unless-stopped \
  --network herdsync-net \
  -p 8000:8000 \
  -v auth-keys:/app/keys \
  --env-file /home/ubuntu/goat-api.env \
  $IMAGE

echo "Waiting for goat-api service to start..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "goat-api service is healthy"
    exit 0
  fi
  echo "  attempt $i/30 — not ready yet"
  sleep 2
done
echo "goat-api service failed to start within 60s"
docker logs goat-api --tail 30
exit 1
