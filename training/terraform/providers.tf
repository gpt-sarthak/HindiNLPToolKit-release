terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.0" }
  }
}

# Credentials come from your environment (AWS_PROFILE, or the default profile
# set up by `aws configure`). Nothing account-specific lives in these files.
provider "aws" {
  region = var.region
}
