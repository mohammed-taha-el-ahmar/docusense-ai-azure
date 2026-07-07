variable "resource_group_name" { type = string }
variable "location" { type = string }
variable "name" { type = string }
variable "tags" { type = map(string) }

resource "azurerm_cognitive_account" "content_safety" {
  name                  = var.name
  resource_group_name   = var.resource_group_name
  location              = var.location
  kind                  = "ContentSafety"
  sku_name              = "S0"
  custom_subdomain_name = var.name
  tags                  = var.tags
}

output "endpoint" {
  value = azurerm_cognitive_account.content_safety.endpoint
}

output "content_safety_id" {
  value = azurerm_cognitive_account.content_safety.id
}
