# deploy.ps1 — Windows twin of deploy.sh: build the web-app image, ship it to
# one EC2 instance via S3, and start it. Prints http://<public-ip>/ at the end.
#
#   .\deploy\deploy.ps1               # build + package + terraform apply
#   .\deploy\deploy.ps1 -SkipBuild    # reuse the existing local image
#
# Needs Docker Desktop running, terraform >= 1.5, AWS CLI v2 and AWS credentials
# for your own account (README, "AWS credentials"). The trained models must be
# in scoring\models\ first (training\README.md). Running costs ~$0.17/hr; use
# stop.ps1 between sessions and destroy.ps1 when done.
param([switch]$SkipBuild)
$ErrorActionPreference = "Stop"

$Dir   = $PSScriptRoot
$Root  = Split-Path -Parent $Dir
$Build = Join-Path $Dir "build"
$Tar   = Join-Path $Build "image.tar"
$Gz    = Join-Path $Build "image.tar.gz"
$New   = "$Gz.new"
$Image = "hindi-nlp-toolkit:latest"
New-Item -ItemType Directory -Force $Build | Out-Null

$required = "base_model_wiki50m.pt", "vocab.pkl", "hdtb_dsps_grammar", "recommender_weights.json",
    "trigram_katz\meta.json", "trigram_katz\vocab.json", "trigram_katz\uni_logp.npy",
    "trigram_katz\bi_key.npy", "trigram_katz\bi_logp.npy", "trigram_katz\bi_alpha_log.npy",
    "trigram_katz\tri_key.npy", "trigram_katz\tri_logp.npy", "trigram_katz\tri_ctx_key.npy",
    "trigram_katz\tri_alpha_log.npy"
$missing = $required | Where-Object { -not (Test-Path (Join-Path $Root "scoring\models\$_")) }
if ($missing) {
    $missing | ForEach-Object { Write-Host "missing model file: scoring\models\$_" }
    throw "train or copy the models first (training\README.md)"
}
aws sts get-caller-identity --query Account --output text | Out-Null
if ($LASTEXITCODE -ne 0) { throw "AWS credentials not working - run 'aws configure' (see README)" }

if (-not $SkipBuild) {
    Write-Host "==> [1/3] docker build (~10-25 min the first time)"
    docker build --provenance=false -t $Image $Root
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

Write-Host "==> [2/3] docker save + gzip -> build\image.tar.gz"
# The gzip goes to a .new name and replaces image.tar.gz only when complete, so
# an interrupted run never leaves a truncated tarball for terraform to upload.
foreach ($stale in @($Tar, $New)) { if (Test-Path $stale) { Remove-Item $stale -Force } }
docker save $Image -o $Tar
if ($LASTEXITCODE -ne 0) { throw "docker save failed" }
$fin  = [System.IO.File]::OpenRead($Tar)
$fout = [System.IO.File]::Create($New)
$gzip = New-Object System.IO.Compression.GZipStream($fout, [System.IO.Compression.CompressionLevel]::Fastest)
try { $fin.CopyTo($gzip) } finally { $gzip.Dispose(); $fout.Dispose(); $fin.Dispose() }
Move-Item -Path $New -Destination $Gz -Force
Remove-Item $Tar -Force
Write-Host ("    {0:N2} GB" -f ((Get-Item $Gz).Length / 1GB))

Write-Host "==> [3/3] terraform apply (uploads the image to S3 and launches the instance)"
Push-Location (Join-Path $Dir "terraform")
try {
    terraform init -input=false | Out-Null
    terraform apply
    if ($LASTEXITCODE -ne 0) { throw "terraform apply failed" }
    $ip  = terraform output -raw public_ip
    $iid = terraform output -raw instance_id
} finally { Pop-Location }

Write-Host ""
Write-Host "=================================================================="
Write-Host " Instance $iid is booting. The app is live in ~5 min at:"
Write-Host "     http://$ip/"
Write-Host "------------------------------------------------------------------"
Write-Host " Boot log : aws ec2 get-console-output --instance-id $iid --query Output --output text"
Write-Host " Shell    : aws ssm start-session --target $iid"
Write-Host " Pause    : .\deploy\stop.ps1      (keeps the disk, ~`$2.40/month)"
Write-Host " Resume   : .\deploy\start.ps1     (prints the new IP)"
Write-Host " Remove   : .\deploy\destroy.ps1   (~`$0 afterwards)"
Write-Host "=================================================================="
