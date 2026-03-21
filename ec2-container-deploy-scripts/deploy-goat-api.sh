#!/bin/bash
set -e

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

sleep 8
curl -f http://localhost:8000/health