output "alb_dns_name" {
  description = "Public Application Load Balancer DNS entrypoint"
  value       = aws_lb.api.dns_name
}

output "s3_bucket_name" {
  description = "S3 bucket for document images and model artifacts"
  value       = aws_s3_bucket.artifacts.id
}

output "rds_endpoint" {
  description = "PostgreSQL RDS connection endpoint"
  value       = aws_db_instance.postgres.endpoint
}

output "ecs_cluster_name" {
  description = "ECS cluster identifier"
  value       = aws_ecs_cluster.main.name
}
