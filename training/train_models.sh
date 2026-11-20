#!/usr/bin/env bash
# train_models.sh — train every model on one AWS GPU instance, then fetch them
# into scoring/models/ and tear the stack down.
#
#   bash training/train_models.sh                          # all steps
#   TF_VAR_steps="pcfg" bash training/train_models.sh      # a subset
#   TF_VAR_instance_type=g5.xlarge bash training/train_models.sh
#
# Needs: terraform >= 1.5, the AWS CLI v2, git, and AWS credentials for your own
# account (see the README, "AWS credentials"). THIS SPENDS MONEY: ~$1-2 for a
# full run on g4dn.xlarge. Terraform shows its plan and asks before creating
# anything, and again before destroying.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF="$ROOT/training/terraform"
BUILD="$TF/build"
mkdir -p "$BUILD"

for tool in terraform aws git tar; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
aws sts get-caller-identity --query Account --output text >/dev/null \
  || { echo "AWS credentials not working — run 'aws configure' (see README)" >&2; exit 1; }

echo "==> packing the repository (tracked files, working-tree versions)"
( cd "$ROOT" && git ls-files -z | tar --null -T - -czf "$BUILD/code.tar.gz" )
ls -la "$BUILD/code.tar.gz"

cd "$TF"
terraform init -input=false >/dev/null
echo "==> terraform apply (review the plan, then type 'yes')"
terraform apply
BUCKET="$(terraform output -raw bucket)"
REGION="$(terraform output -raw region)"
IID="$(terraform output -raw instance_id)"
echo "==> instance $IID is training; log: s3://$BUCKET/logs/train.log"

while true; do
  if STATUS="$(aws s3 cp "s3://$BUCKET/DONE" - --region "$REGION" 2>/dev/null)"; then
    break
  fi
  if aws s3 cp "s3://$BUCKET/logs/train.log" "$BUILD/train.log" --region "$REGION" >/dev/null 2>&1; then
    echo "--- $(date +%H:%M:%S) ---"; tail -n 3 "$BUILD/train.log"
  else
    echo "--- $(date +%H:%M:%S) --- waiting for the instance to boot"
  fi
  sleep 120
done
echo "==> job finished: $STATUS"
aws s3 cp "s3://$BUCKET/logs/train.log" "$BUILD/train.log" --region "$REGION" >/dev/null || true

if aws s3 cp "s3://$BUCKET/out/models.tar.gz" "$BUILD/models.tar.gz" --region "$REGION"; then
  DEST="$ROOT/scoring/models"
  if [ -n "$(ls -A "$DEST" 2>/dev/null | grep -v recommender_weights.json || true)" ]; then
    BAK="$ROOT/scoring/models.bak-$(date +%Y%m%d-%H%M%S)"
    echo "==> existing models moved to $BAK"
    mv "$DEST" "$BAK"
    mkdir -p "$DEST"
    cp "$BAK/recommender_weights.json" "$DEST/" 2>/dev/null || true
  fi
  tar -xzf "$BUILD/models.tar.gz" -C "$DEST"
  echo "==> models in $DEST:"; ls -la "$DEST"
else
  echo "no models.tar.gz was published — see $BUILD/train.log" >&2
fi

echo "==> terraform destroy (type 'yes' to remove the bucket, role and security group)"
terraform destroy
case "$STATUS" in OK*) exit 0;; *) exit 1;; esac
