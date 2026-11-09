variable "aws_region" {
  description = "AWS deployment region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "signature-verify"
}

variable "db_password" {
  description = "RDS PostgreSQL master password"
  type        = string
  sensitive   = true
  default     = "ChangeMeInProduction123!"
}

variable "mq_password" {
  description = "Amazon MQ RabbitMQ password"
  type        = string
  sensitive   = true
  default     = "ChangeMeRabbit123!"
}
