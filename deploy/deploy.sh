#!/usr/bin/env bash
# deploy.sh — build the web-app image, ship it to one EC2 instance via S3, and
# start it. Prints http://<public-ip>/ at the end.
#
#   bash deploy/deploy.sh                 # build + package + terraform apply
#   SKIP_BUILD=1 bash deploy/deploy.sh    # reuse the existing local image
#
# Needs: Docker running, terraform >= 1.5, AWS CLI v2, and AWS credentials for
# your own account (README, "AWS credentials"). The trained models must be in
# scoring/models/ first (training/README.md) — they are baked into the image.
# Running costs ~$0.17/hr; use stop.sh between sessions, destroy.sh when done.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DIR")"
BUILD="$DIR/build"
IMAGE="hindi-nlp-toolkit:latest"
mkdir -p "$BUILD"

missing=0
for f in base_model_wiki50m.pt vocab.pkl hdtb_dsps_grammar recommender_weights.json \
         trigram_katz/meta.json trigram_katz/vocab.json trigram_katz/uni_logp.npy \
         trigram_katz/bi_key.npy trigram_katz/bi_logp.npy trigram_katz/bi_alpha_log.npy \
         trigram_katz/tri_key.npy trigram_katz/tri_logp.npy trigram_katz/tri_ctx_key.npy \
         trigram_katz/tri_alpha_log.npy; do
  [ -s "$ROOT/scoring/models/$f" ] || { echo "missing model file: scoring/models/$f" >&2; missing=1; }
done
[ $missing -eq 0 ] || { echo "train or copy the models first (training/README.md)" >&2; exit 1; }
aws sts get-caller-identity --query Account --output text >/dev/null \
  || { echo "AWS credentials not working — run 'aws configure' (see README)" >&2; exit 1; }

if [ -z "${SKIP_BUILD:-}" ]; then
  echo "==> [1/3] docker build (~10-25 min the first time)"
  docker build --provenance=false -t "$IMAGE" "$ROOT"
fi

echo "==> [2/3] docker save | gzip -> build/image.tar.gz"
# Written to a temporary name and moved into place only when complete, so an
# interrupted run can never leave a truncated tarball for terraform to upload.
rm -f "$BUILD/image.tar.gz.new"
docker save "$IMAGE" | gzip -1 > "$BUILD/image.tar.gz.new"
mv -f "$BUILD/image.tar.gz.new" "$BUILD/image.tar.gz"
ls -la "$BUILD/image.tar.gz"

echo "==> [3/3] terraform apply (uploads the image to S3 and launches the instance)"
cd "$DIR/terraform"
terraform init -input=false >/dev/null
terraform apply
IP="$(terraform output -raw public_ip)"
IID="$(terraform output -raw instance_id)"
cat <<EOF

==================================================================
 Instance $IID is booting. The app is live in ~5 min at:
     http://$IP/
------------------------------------------------------------------
 Boot log : aws ec2 get-console-output --instance-id $IID --query Output --output text
 Shell    : aws ssm start-session --target $IID
 Pause    : bash deploy/stop.sh      (keeps the disk, ~\$2.40/month)
 Resume   : bash deploy/start.sh     (prints the new IP)
 Remove   : bash deploy/destroy.sh   (~\$0 afterwards)
==================================================================
EOF
