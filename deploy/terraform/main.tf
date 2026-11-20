locals {
  tags = {
    Project   = var.project
    ManagedBy = "terraform"
    Component = "webapp-demo"
  }

  # MD5 of the local image tarball, computed at plan time. Both the S3 object's
  # change-detection hash and the instance's user_data derive from THIS, never
  # from aws_s3_object.image.etag: S3 returns a *multipart* ETag ("<md5>-<parts>")
  # for a 1.7 GB upload, so the attribute's real value only becomes known during
  # apply. Interpolating it into user_data made the value change mid-apply and
  # Terraform aborted with "Provider produced inconsistent final plan".
  image_md5 = filemd5("${path.module}/${var.image_tar}")
}

resource "random_id" "suffix" {
  byte_length = 4
}

# ---------- networking: default VPC / first default subnet ----------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---------- AMI: latest Amazon Linux 2023 (x86_64) ----------
# AL2023 ships awscli v2 + the SSM agent preinstalled; Docker is one dnf away.
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-kernel-6.1-x86_64"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ---------- S3 bucket holding the docker image tarball ----------
resource "aws_s3_bucket" "app" {
  bucket        = "${var.project}-${random_id.suffix.hex}"
  force_destroy = true # destroy removes the image object too
  tags          = local.tags
}

resource "aws_s3_object" "image" {
  bucket = aws_s3_bucket.app.id
  key    = "image.tar.gz"
  source = "${path.module}/${var.image_tar}"

  # source_hash, not etag: it triggers a re-upload when the local tarball changes
  # without being compared against S3's multipart ETag, which never equals a plain
  # MD5. With etag here, every apply saw a bogus diff and re-uploaded 1.7 GB.
  source_hash = local.image_md5
}

# ---------- IAM: instance role (read image from S3 + SSM shell) ----------
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "app" {
  name               = "${var.project}-${random_id.suffix.hex}"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "s3_read" {
  name = "s3-image-read"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.app.arn}/*" },
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.app.arn }
    ]
  })
}

# SSM Session Manager: shell access without opening port 22 or managing keys.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.project}-${random_id.suffix.hex}"
  role = aws_iam_role.app.name
}

# ---------- security group: inbound HTTP only ----------
resource "aws_security_group" "app" {
  name        = "${var.project}-${random_id.suffix.hex}"
  description = "hindi-nlp webapp: inbound HTTP only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "all outbound (S3, dnf, SSM, Stanza download)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# ---------- EC2 instance ----------
resource "aws_instance" "app" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.app.name
  associate_public_ip_address = true

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    bucket     = aws_s3_bucket.app.id
    region     = var.region
    image_etag = local.image_md5
  })
  # Recreate the instance when the image (hence user_data) changes, so a rebuilt
  # image is actually loaded on the next deploy script without a manual destroy.
  user_data_replace_on_change = true

  tags = merge(local.tags, { Name = var.project })

  depends_on = [aws_s3_object.image, aws_iam_role_policy.s3_read]
}
