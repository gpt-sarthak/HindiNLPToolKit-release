#!/usr/bin/env bash
# start.sh — start a stopped instance again. The container restarts on its own;
# the public IP is new each time (no Elastic IP), so the new URL is printed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/terraform"
IID="$(terraform output -raw instance_id)"
REGION="$(terraform output -raw region)"
echo "==> starting $IID in $REGION"
aws ec2 start-instances --instance-ids "$IID" --region "$REGION" >/dev/null
aws ec2 wait instance-running --instance-ids "$IID" --region "$REGION"
IP="$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "running — app back in ~5 min at http://$IP/"
