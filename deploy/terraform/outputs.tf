output "public_ip" {
  description = "Public IPv4 of the instance. Open http://<public_ip>/ (allow ~5-10 min after deploy for image load + model warmup)."
  value       = aws_instance.app.public_ip
}

output "url" {
  description = "Convenience URL."
  value       = "http://${aws_instance.app.public_ip}/"
}

output "instance_id" {
  description = "EC2 instance id (used by the stop / start scripts)."
  value       = aws_instance.app.id
}

output "bucket" {
  description = "S3 bucket holding the docker image tarball."
  value       = aws_s3_bucket.app.id
}

output "region" {
  description = "Region the stack is in (used by the stop / start scripts)."
  value       = var.region
}
