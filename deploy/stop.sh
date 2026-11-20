#!/usr/bin/env bash
# stop.sh — stop the instance between sessions. Compute billing pauses; only the
# 30 GB disk keeps costing (~$2.40/month). Resume with start.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/terraform"
IID="$(terraform output -raw instance_id)"
REGION="$(terraform output -raw region)"
echo "==> stopping $IID in $REGION"
aws ec2 stop-instances --instance-ids "$IID" --region "$REGION" >/dev/null
echo "stopping. Resume with deploy/start.sh (the public IP will change)."
