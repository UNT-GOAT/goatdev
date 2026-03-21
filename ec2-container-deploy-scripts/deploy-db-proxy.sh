#!/bin/bash
set -e

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

sleep 8
curl -f http://localhost:8003/health