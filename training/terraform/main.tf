###############################################################################
# Model-training stack.
#
# One EC2 instance boots, downloads the repository tarball from its own S3
# bucket, runs training/run_all.sh, uploads models.tar.gz + logs + a DONE
# marker, and terminates itself. Shares nothing with deploy/terraform: its own
# state, bucket, IAM role and security group (no inbound ports; shell access
# is via SSM Session Manager).
###############################################################################

locals {
  name = "${var.project}-${random_id.suffix.hex}"
  tags = {
    Project   = var.project
    ManagedBy = "terraform"
  }
}

resource "random_id" "suffix" {
  byte_length = 4
}

# --- network: default VPC ----------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# --- AMI: Deep Learning AMI (PyTorch + NVIDIA driver) ------------------------
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# --- storage -----------------------------------------------------------------
resource "aws_s3_bucket" "work" {
  bucket        = local.name
  force_destroy = true # destroy also removes the uploaded code and outputs
  tags          = local.tags
}

resource "aws_s3_bucket_public_access_block" "work" {
  bucket                  = aws_s3_bucket.work.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "code" {
  bucket      = aws_s3_bucket.work.id
  key         = "code/code.tar.gz"
  source      = var.code_archive
  source_hash = filemd5(var.code_archive)
}

# --- identity ----------------------------------------------------------------
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "job" {
  name               = local.name
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "bucket" {
  name = "${local.name}-bucket"
  role = aws_iam_role.job.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.work.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.work.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.job.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "job" {
  name = local.name
  role = aws_iam_role.job.name
  tags = local.tags
}

resource "aws_security_group" "job" {
  name        = local.name
  description = "Model training job - egress only, no inbound"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# --- the instance ------------------------------------------------------------
resource "aws_instance" "job" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.job.id]
  iam_instance_profile   = aws_iam_instance_profile.job.name

  # A `shutdown -h` from inside the box terminates it, so the dead-man timer
  # and the final shutdown really stop the billing.
  instance_initiated_shutdown_behavior = "terminate"

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    bucket           = aws_s3_bucket.work.id
    region           = var.region
    steps            = var.steps
    wiki_date        = var.wiki_date
    lstm_args        = var.lstm_args
    dead_man_minutes = var.dead_man_minutes
  })
  user_data_replace_on_change = true

  depends_on = [aws_s3_object.code, aws_iam_role_policy.bucket]

  tags = merge(local.tags, { Name = local.name })
}
