# destroy.ps1 — remove everything deploy.ps1 created (instance, disk, S3 bucket
# + image, IAM role, security group). Idle cost afterwards is ~$0.
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "terraform")
try {
    terraform destroy
    if ($LASTEXITCODE -ne 0) { throw "terraform destroy failed - check the AWS console for leftovers" }
} finally { Pop-Location }
Write-Host ""
Write-Host "==> destroyed. Double-check nothing is left:"
Write-Host "    aws ec2 describe-instances --filters Name=tag:Project,Values=hindi-nlp Name=instance-state-name,Values=pending,running,stopping,stopped --query `"Reservations[].Instances[].InstanceId`""
