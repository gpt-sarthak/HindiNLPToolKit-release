variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type. t3.xlarge = 4 vCPU / 16 GB, ~$0.17/hr in us-east-1 —
    enough for all five scorers (Java for the PCFG, Stanza ~1 GB, torch).
    Switch to m5.xlarge for steady, non-burstable CPU if the one-time model
    warm-up runs out of burst credits.
  EOT
  type        = string
  default     = "t3.xlarge"
}

variable "project" {
  description = "Name prefix / tag applied to all resources."
  type        = string
  default     = "hindi-nlp"
}

variable "image_tar" {
  description = "Path (relative to this terraform dir) to the gzipped docker image tarball produced by deploy.sh / deploy.ps1."
  type        = string
  default     = "../build/image.tar.gz"
}

variable "root_volume_gb" {
  description = "Root EBS (gp3) size in GB. Needs room for the ~2 GB tarball + ~3.5 GB loaded image + Stanza cache."
  type        = number
  default     = 30
}
