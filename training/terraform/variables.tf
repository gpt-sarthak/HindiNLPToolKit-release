variable "region" {
  description = "AWS region for the training instance and its bucket."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix and Project tag. Kept distinct from the web app's deploy stack."
  type        = string
  default     = "hindi-nlp-train"
}

variable "instance_type" {
  description = <<-EOT
    g4dn.xlarge = 1x NVIDIA T4 (the GPU the shipped LSTM was trained on),
    4 vCPU, 16 GB RAM, ~$0.53/hr on demand in us-east-1. Needs a non-zero
    "Running On-Demand G and VT instances" vCPU quota (see training/README.md).
    A CPU type (e.g. c7i.4xlarge) also works, but the LSTM step then sees far
    fewer training steps inside its 25-minute budget.
  EOT
  type        = string
  default     = "g4dn.xlarge"
}

variable "root_volume_gb" {
  description = "The Deep Learning AMI is large; leave room for it plus ~3 GB of Wikipedia text."
  type        = number
  default     = 150
}

variable "steps" {
  description = "Space-separated training/run_all.sh steps: wiki vocab trigram lstm pcfg."
  type        = string
  default     = "wiki vocab trigram lstm pcfg"
}

variable "wiki_date" {
  description = "Hindi Wikipedia dump date (e.g. 20260201) or 'latest'."
  type        = string
  default     = "latest"
}

variable "lstm_args" {
  description = "Extra arguments for training/lstm/train_lstm.py, e.g. \"--minutes 60\"."
  type        = string
  default     = ""
}

variable "dead_man_minutes" {
  description = <<-EOT
    Cost cap: a `shutdown -h +N` armed at boot whatever happens; with
    shutdown behaviour = terminate this bounds the worst case to ~N minutes.
    The full chain takes ~2 hours on g4dn.xlarge.
  EOT
  type        = number
  default     = 240
}

variable "code_archive" {
  description = "Path of the code tarball built by train_models.sh / .ps1."
  type        = string
  default     = "build/code.tar.gz"
}

variable "ami_name_filter" {
  description = <<-EOT
    Deep Learning AMI name glob (PyTorch + NVIDIA driver preinstalled). Amazon
    renames these now and then; if `plan` finds no AMI, widen the glob rather
    than pinning an id (ids are per region and go stale).
  EOT
  type        = string
  default     = "Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*"
}
