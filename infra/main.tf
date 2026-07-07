locals {
  name_prefix  = "${var.project}-${var.environment}"
  short_prefix = replace("${var.project}${var.environment}", "-", "")
  tags = merge(var.tags, {
    environment = var.environment
  })
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name_prefix}"
  location = var.location
  tags     = local.tags
}

resource "random_string" "suffix" {
  length  = 5
  upper   = false
  special = false
  numeric = true
}

module "storage" {
  source              = "./modules/storage"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name                = "st${local.short_prefix}${random_string.suffix.result}"
  tags                = local.tags
}

module "keyvault" {
  source              = "./modules/keyvault"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name                = "kv-${local.name_prefix}-${random_string.suffix.result}"
  tags                = local.tags
}

module "monitoring" {
  source              = "./modules/monitoring"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  tags                = local.tags
}

module "openai" {
  source                     = "./modules/openai"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  name                       = "aoai-${local.name_prefix}-${random_string.suffix.result}"
  chat_deployment_name       = var.openai_chat_deployment
  embedding_deployment_name  = var.openai_embedding_deployment
  tags                       = local.tags
}

module "ai_search" {
  source              = "./modules/ai_search"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name                = "srch-${local.name_prefix}-${random_string.suffix.result}"
  tags                = local.tags
}

module "content_safety" {
  source              = "./modules/content_safety"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name                = "cs-${local.name_prefix}-${random_string.suffix.result}"
  tags                = local.tags
}

module "aml_workspace" {
  source                  = "./modules/aml_workspace"
  resource_group_name     = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  name                    = "aml-${local.name_prefix}"
  storage_account_id      = module.storage.storage_account_id
  key_vault_id            = module.keyvault.key_vault_id
  application_insights_id = module.monitoring.application_insights_id
  tags                    = local.tags
}
