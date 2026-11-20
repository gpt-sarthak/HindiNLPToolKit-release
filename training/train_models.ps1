# train_models.ps1 — Windows twin of train_models.sh: train every model on one
# AWS GPU instance, fetch them into scoring\models\, tear the stack down.
#
#   .\training\train_models.ps1
#   $env:TF_VAR_steps = "pcfg"; .\training\train_models.ps1
#
# Needs terraform >= 1.5, AWS CLI v2, git and tar (built into Windows 10+), and
# AWS credentials for your own account (README, "AWS credentials").
# THIS SPENDS MONEY (~$1-2 for a full run). Terraform asks before it creates or
# destroys anything.
$ErrorActionPreference = "Stop"
$Root  = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Tf    = Join-Path $Root "training\terraform"
$Build = Join-Path $Tf "build"
New-Item -ItemType Directory -Force $Build | Out-Null

foreach ($tool in "terraform", "aws", "git", "tar") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "missing tool: $tool" }
}
aws sts get-caller-identity --query Account --output text | Out-Null
if ($LASTEXITCODE -ne 0) { throw "AWS credentials not working - run 'aws configure' (see README)" }

Write-Host "==> packing the repository (tracked files, working-tree versions)"
$list = Join-Path $Build "files.txt"
Push-Location $Root
try {
    # LF-separated list: bsdtar would keep a CR in every file name otherwise.
    [IO.File]::WriteAllText($list, ((git ls-files) -join "`n") + "`n")
    tar -czf (Join-Path $Build "code.tar.gz") -T $list
    if ($LASTEXITCODE -ne 0) { throw "tar failed" }
} finally { Pop-Location }

Push-Location $Tf
try {
    terraform init -input=false | Out-Null
    Write-Host "==> terraform apply (review the plan, then type 'yes')"
    terraform apply
    if ($LASTEXITCODE -ne 0) { throw "terraform apply failed" }
    $bucket = terraform output -raw bucket
    $region = terraform output -raw region
    $iid    = terraform output -raw instance_id
    Write-Host "==> instance $iid is training; log: s3://$bucket/logs/train.log"

    $log = Join-Path $Build "train.log"
    while ($true) {
        $status = aws s3 cp "s3://$bucket/DONE" - --region $region 2>$null
        if ($LASTEXITCODE -eq 0 -and $status) { break }
        aws s3 cp "s3://$bucket/logs/train.log" $log --region $region 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "--- $(Get-Date -Format HH:mm:ss) ---"; Get-Content $log -Tail 3
        } else {
            Write-Host "--- $(Get-Date -Format HH:mm:ss) --- waiting for the instance to boot"
        }
        Start-Sleep -Seconds 120
    }
    Write-Host "==> job finished: $status"
    aws s3 cp "s3://$bucket/logs/train.log" $log --region $region | Out-Null

    $tgz = Join-Path $Build "models.tar.gz"
    aws s3 cp "s3://$bucket/out/models.tar.gz" $tgz --region $region
    if ($LASTEXITCODE -eq 0) {
        $dest = Join-Path $Root "scoring\models"
        $existing = Get-ChildItem $dest -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne "recommender_weights.json" }
        if ($existing) {
            $bak = Join-Path $Root ("scoring\models.bak-" + (Get-Date -Format yyyyMMdd-HHmmss))
            Write-Host "==> existing models moved to $bak"
            Move-Item $dest $bak
            New-Item -ItemType Directory -Force $dest | Out-Null
            Copy-Item (Join-Path $bak "recommender_weights.json") $dest -ErrorAction SilentlyContinue
        }
        tar -xzf $tgz -C $dest
        Write-Host "==> models in ${dest}:"; Get-ChildItem $dest | Format-Table Name, Length -AutoSize
    } else {
        Write-Warning "no models.tar.gz was published - see $log"
    }

    Write-Host "==> terraform destroy (type 'yes' to remove the bucket, role and security group)"
    terraform destroy
    if (-not "$status".StartsWith("OK")) { exit 1 }
} finally { Pop-Location }
