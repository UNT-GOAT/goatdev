#!/bin/bash
set -e

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

sleep 8
curl -f http://localhost:8001/auth/health