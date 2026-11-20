#!/usr/bin/env bash
# destroy.sh — remove everything deploy.sh created (instance, disk, S3 bucket +
# image, IAM role, security group). Idle cost afterwards is ~$0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/terraform"
terraform destroy
echo
echo "==> destroyed. Double-check nothing is left:"
echo "    aws ec2 describe-instances --filters Name=tag:Project,Values=hindi-nlp \\"
echo "        Name=instance-state-name,Values=pending,running,stopping,stopped \\"
echo "        --query 'Reservations[].Instances[].InstanceId'"
