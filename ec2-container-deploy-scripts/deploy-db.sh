#!/bin/bash
set -e

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 937249941844.dkr.ecr.us-east-2.amazonaws.com


IMAGE=937249941844.dkr.ecr.us-east-2.amazonaws.com/herdsync-db:latest

docker pull $IMAGE
docker rm -f db || true

docker run -d \
  --name db \
  --restart unless-stopped \
  -p 8002:8002 \
  --env-file /home/ubuntu/db.env \
  $IMAGE

sleep 8
curl -f http://localhost:8002/health
