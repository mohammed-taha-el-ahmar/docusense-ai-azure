variable "environment" {
  description = "Deployment environment (dev, prod)."
  type        = string
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be one of: dev, prod."
  }
}

variable "location" {
  description = "Azure region — pick one that supports Azure OpenAI + AI Search."
  type        = string
  default     = "swedencentral"
}

variable "project" {
  description = "Project short name."
  type        = string
  default     = "docusense"
}

variable "openai_chat_deployment" {
  description = "Azure OpenAI chat model deployment (e.g. gpt-5.1)."
  type        = string
  default     = "gpt-5.1"
}

variable "openai_embedding_deployment" {
  description = "Azure OpenAI embedding deployment name."
  type        = string
  default     = "text-embedding-3-large"
}

variable "tags" {
  type = map(string)
  default = {
    project    = "docusense"
    managed_by = "terraform"
    owner      = "melar"
  }
}
