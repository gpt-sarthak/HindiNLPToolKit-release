# deploy/ — run the web app on one AWS EC2 instance

```
deploy.sh / deploy.ps1     build the image, gzip it, terraform apply, print the URL
stop.sh   / stop.ps1       stop the instance (disk kept)
start.sh  / start.ps1      start it again (prints the new public IP)
destroy.sh/ destroy.ps1    remove everything
iam-policy.json            least-privilege policy for the IAM user that runs these
terraform/                 the stack (main.tf, variables.tf, outputs.tf, user_data.sh.tftpl)
build/                     image.tar.gz (git-ignored, ~1.4 GB)
```

## Before the first deploy

1. The trained models are in `scoring/models/` (see `training/README.md`) —
   the deploy script refuses to run without them, because they are baked into
   the image.
2. Docker is running; `terraform` (≥ 1.5) and the AWS CLI v2 are on `PATH`.
3. AWS credentials for your own account work: `aws sts get-caller-identity`.
   The README (§5b) walks through creating an IAM user with `iam-policy.json`.

## What `deploy` creates (us-east-1 by default)

- an **S3 bucket** `hindi-nlp-<random>` holding the gzipped Docker image
  (`docker save` → S3 → `docker load`; no registry needed);
- an **IAM role + instance profile** allowing the instance to read that bucket
  and to accept SSM sessions;
- a **security group** with inbound **port 80 only** (no SSH);
- one **`t3.xlarge`** (4 vCPU, 16 GB) on Amazon Linux 2023 with a 30 GB disk.
  At boot it installs Docker, loads the image and runs
  `docker run -d --restart unless-stopped -p 80:8001 --name hindi-nlp …`.

The app answers at `http://<public-ip>/` about 5 minutes after `apply`
(image load + model warm-up). Plain HTTP, no domain, no Elastic IP — the IP
changes on every start.

Re-running `deploy` after a code change rebuilds the image; the new image changes
the instance's boot script, so Terraform replaces the instance and the new image
is loaded. `SKIP_BUILD=1` (`-SkipBuild`) re-packages the existing local image.

## Settings

Terraform variables, set as environment variables before `deploy`:

| Variable | Default | |
|---|---|---|
| `TF_VAR_region` | `us-east-1` | |
| `TF_VAR_instance_type` | `t3.xlarge` | `m5.xlarge` for steady (non-burstable) CPU |
| `TF_VAR_project` | `hindi-nlp` | name prefix of every resource; keep the `hindi-nlp` prefix or widen `iam-policy.json` |
| `TF_VAR_root_volume_gb` | `30` | |

## Cost (us-east-1, on demand)

| State | Cost |
|---|---|
| running | ~$0.17/hour |
| stopped | ~$2.40/month (the 30 GB disk) + ~$0.04 (S3) |
| destroyed | ~$0 |
| ⚠ left running for a month | ~$124 — always stop or destroy |

## Operating it

- Boot log: `aws ec2 get-console-output --instance-id <id> --query Output --output text`
- Shell (no SSH needed): `aws ssm start-session --target <id>`, then
  `docker logs -f hindi-nlp`
- A stuck job can be cancelled from the page; restarting the container
  (`docker restart hindi-nlp`) also clears the in-memory job queue.
- `terraform/terraform.tfstate` records what was created in your account — keep
  it (it is git-ignored) or `destroy` cannot find the resources.
