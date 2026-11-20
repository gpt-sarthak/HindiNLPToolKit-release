# stop.ps1 — stop the instance between sessions. Compute billing pauses; only
# the 30 GB disk keeps costing (~$2.40/month). Resume with start.ps1.
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "terraform")
try {
    $iid    = terraform output -raw instance_id
    $region = terraform output -raw region
} finally { Pop-Location }
if (-not $iid) { throw "no deployed stack found - run deploy.ps1 first" }

Write-Host "==> stopping $iid in $region"
aws ec2 stop-instances --instance-ids $iid --region $region | Out-Null
Write-Host "stopping. Resume with .\deploy\start.ps1 (the public IP will change)."
