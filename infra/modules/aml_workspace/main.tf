variable "resource_group_name" { type = string }
variable "location" { type = string }
variable "name" { type = string }
variable "storage_account_id" { type = string }
variable "key_vault_id" { type = string }
variable "application_insights_id" { type = string }
variable "tags" { type = map(string) }

resource "azurerm_machine_learning_workspace" "this" {
  name                          = var.name
  resource_group_name           = var.resource_group_name
  location                      = var.location
  storage_account_id            = var.storage_account_id
  key_vault_id                  = var.key_vault_id
  application_insights_id       = var.application_insights_id
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

resource "azurerm_machine_learning_compute_cluster" "cpu" {
  name                          = "cpu-cluster"
  location                      = var.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.this.id
  vm_priority                   = "Dedicated"
  vm_size                       = "Standard_DS3_v2"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT120S"
  }

  identity {
    type = "SystemAssigned"
  }
}

output "workspace_name" {
  value = azurerm_machine_learning_workspace.this.name
}

output "workspace_id" {
  value = azurerm_machine_learning_workspace.this.id
}
