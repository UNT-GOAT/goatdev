# Goatdev - AWS Architecture - Current Headway

## Architecture Components

### 1. Image Capture Layer (On-Premises)

**Raspberry Pi (Facility)**

- Location: On-site at the Minnesota goat facility
- Function: Captures images from 3 USB cameras (side, top, front views)
- Connectivity: Uploads directly to S3 via AWS SDK
- Authentication: IAM user `goat-pi-capture` with minimal permissions (S3 PutObject only)
- Naming Convention: Images uploaded as `captures/{timestamp}/{timestamp}_{view}.jpg`

### 2. Storage Layer (Amazon S3)

**goat-captures-{ACCOUNT_ID}**

- Purpose: Raw image storage from Raspberry Pi
- Access: Write from Pi, Read from EC2
- Lifecycle Policy:
  - Transition to Glacier: 90 days
  - Expiration: 365 days
- Security: Private, no public access

**goat-processed-{ACCOUNT_ID}**

- Purpose: Annotated images and JSON results from processing
- Access: Write from EC2, Read from mobile app
- Lifecycle Policy:
  - Transition to Glacier: 90 days
  - Versioning: Enabled
- Security: Private, presigned URLs for access

**goat-web-{ACCOUNT_ID}**

- Purpose: Static web application hosting
- Access: Public via CloudFront
- Security: Origin Access Identity (OAI) restricts direct S3 access

### 3. Compute Layer (Amazon EC2)

**Instance: i-xxxxxxxxxd2cd**

- Type: t3.medium (2 vCPU, 4GB RAM)
- OS: Ubuntu 22.04 LTS
- Storage: 30GB gp3
- Availability Zone: us-east-2b
- Elastic IP: x.x.x.x

**Container Runtime**

- Docker Engine running the goat-api container
- Image pulled from Amazon ECR
- Auto-restart policy: `unless-stopped`
- Port mapping: 8000:8000

**Application Stack**

- FastAPI web framework
- Uvicorn ASGI server
- YOLO v8 segmentation models (3 models: side, top, front)
- Python 3.11 runtime

**IAM Role: goat-ec2-role**

- S3 read access to goat-captures bucket
- S3 read/write access to goat-processed bucket
- ECR pull access for container images

### 4. Container Registry (Amazon ECR)

**Repository: goat-api**

- Stores Docker images for the API
- Tagged with git commit SHA and `latest`
- Images built and pushed via GitHub Actions
- ~500MB image size (CPU-only PyTorch)

### 5. Database Layer (Amazon RDS)

**Instance: goat-db**

- Engine: PostgreSQL 15
- Instance Class: db.t3.micro
- Storage: 20GB gp2
- Endpoint: goat-db.xxxxx.us-east-2.rds.amazonaws.com
- Port: 5432
- Multi-AZ: No (single AZ for cost savings)
- Backup Retention: 7 days
- Public Access: No (VPC only)

**Security**

- Security group allows port 5432 only from EC2 security group
- No internet-facing access
- Encrypted at rest

### 6. Content Delivery (Amazon CloudFront)

**Distribution: EXXXXXXXXXXXXX**

- Domain: xxxxxxxxxxxxxx.cloudfront.net
- Origin: goat-web S3 bucket (static website)
- Price Class: PriceClass_100 (North America, Europe)
- HTTPS: Enforced (HTTP redirects to HTTPS)
- Purpose: Serves the tablet/mobile web application

### 7. Networking (Amazon VPC)

**VPC: vpc-xxxxxxxxxxxxx**

- Region: us-east-2 (Ohio)
- Subnets:
  - subnet-xxxxxxxxxxxxx (us-east-2b)
  - subnet-xxxxxxxxxxxxx (us-east-2c)

**Security Groups**

_goat-ec2-sg (sg-xxxxxxxxxdae0)_
| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | 0.0.0.0/0 | SSH access |
| 80 | TCP | 0.0.0.0/0 | HTTP |
| 443 | TCP | 0.0.0.0/0 | HTTPS |
| 8000 | TCP | 0.0.0.0/0 | FastAPI |

_goat-rds-sg (sg-xxxxxxxxxdaeo)_
| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 5432 | TCP | goat-ec2-sg | PostgreSQL from EC2 only |

### 8. Monitoring (Amazon CloudWatch)

**Alarms**

- `goat-ec2-high-cpu`: Triggers when CPU > 80% for 5 minutes
- `goat-ec2-status-check`: Triggers on instance status check failure

**SNS Topic: goat-alerts**

- Email notifications for alarm triggers

### 9. CI/CD Pipeline (GitHub Actions)

**Workflow: deploy-api.yml**

- Trigger: Push to `main` branch (paths: goat-api/**, model/**)
- Build: Creates Docker image from repo root context
- Push: Uploads to ECR with commit SHA tag
- Deploy: SSH to EC2, pull image, restart container
- Verify: Health check after deployment

**Secrets Required**

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- EC2_HOST
- EC2_SSH_KEY

---

## Data Flow

### Image Capture & Upload Flow

```
1. Raspberry Pi captures images from 3 cameras
2. Images named with timestamp: {timestamp}_{side|top|front}.jpg
3. Pi uploads to s3://goat-captures/{timestamp}/
4. Upload authenticated via IAM user credentials
```

### Processing Flow

```
1. Mobile app or scheduler calls POST /process/{timestamp}?goat_id={id}
2. EC2 downloads images from goat-captures bucket
3. YOLO models process each view:
   - Side: head height, withers height, rump height
   - Top: body width
   - Front: body width
4. Annotated images uploaded to goat-processed bucket
5. Results JSON uploaded to goat-processed bucket
6. Measurements saved to RDS PostgreSQL
7. API returns completion status with presigned URLs
```

### Results Retrieval Flow

```
1. Webapp polls GET /results/{timestamp}
2. API returns status (queued/processing/completed/failed)
3. On completion, response includes:
   - Measurements (cm)
   - Confidence scores
   - Presigned URLs for annotated images (1-hour expiry)
```

---

## Security Architecture

### Network Security

- RDS isolated in private subnet, no public access
- EC2 in public subnet with security group restrictions
- All S3 buckets private (no public access)
- CloudFront enforces HTTPS

### Authentication & Authorization

- Pi uses dedicated IAM user with minimal permissions
- EC2 uses IAM role (no stored credentials)
- GitHub Actions uses dedicated deploy IAM user
- RDS credentials stored in EC2 environment file only

### Data Protection

- S3 server-side encryption (SSE-S3)
- RDS encrypted at rest
- HTTPS for all external communications
- Presigned URLs for temporary S3 access

---

## Cost Breakdown (Estimated Monthly)

| Service            | Estimate   |
| ------------------ | ---------- |
| EC2 t3.medium      | $30        |
| RDS db.t3.micro    | $13        |
| S3 + Data Transfer | $5-10      |
| CloudFront         | $1-2       |
| ECR                | $1         |
| **Total**          | **$50-60** |

Budget alert configured at $60/month.

---

## API Endpoints

| Method | Endpoint                          | Purpose                               |
| ------ | --------------------------------- | ------------------------------------- |
| GET    | /health                           | Health check                          |
| POST   | /process/{timestamp}?goat_id={id} | Start async processing                |
| GET    | /results/{timestamp}              | Poll for results                      |
| GET    | /goats                            | List goats (placeholder)              |
| GET    | /goats/{id}                       | Get goat details (placeholder)        |
| GET    | /goats/{id}/measurements          | Get measurement history (placeholder) |

---

## Future Enhancements

- [ ] Route 53 custom domain
- [ ] SSL/TLS certificate for API (requires domain)
- [ ] Wire up RDS database (currently placeholder)
- [ ] Complete Raspberry Pi capture automation
- [ ] Webapp integration
- [ ] CloudWatch Logs integration
