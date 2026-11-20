output "bucket" {
  value = aws_s3_bucket.work.id
}

output "region" {
  value = var.region
}

output "instance_id" {
  value = aws_instance.job.id
}
