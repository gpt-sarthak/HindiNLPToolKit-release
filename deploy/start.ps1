# start.ps1 — start a stopped instance again. The container restarts on its own;
# the public IP is new each time (no Elastic IP), so the new URL is printed.
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "terraform")
try {
    $iid    = terraform output -raw instance_id
    $region = terraform output -raw region
} finally { Pop-Location }
if (-not $iid) { throw "no deployed stack found - run deploy.ps1 first" }

Write-Host "==> starting $iid in $region"
aws ec2 start-instances --instance-ids $iid --region $region | Out-Null
aws ec2 wait instance-running --instance-ids $iid --region $region
$ip = aws ec2 describe-instances --instance-ids $iid --region $region `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text
Write-Host "running - app back in ~5 min at http://$ip/"
