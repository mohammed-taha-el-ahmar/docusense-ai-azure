output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "aml_workspace_name" {
  value = module.aml_workspace.workspace_name
}

output "openai_endpoint" {
  value = module.openai.endpoint
}

output "openai_chat_deployment" {
  value = module.openai.chat_deployment_name
}

output "openai_embedding_deployment" {
  value = module.openai.embedding_deployment_name
}

output "search_endpoint" {
  value = module.ai_search.endpoint
}

output "search_index_name" {
  value = "docusense-clauses"
}

output "content_safety_endpoint" {
  value = module.content_safety.endpoint
}

output "storage_account_name" {
  value = module.storage.storage_account_name
}

output "key_vault_name" {
  value = module.keyvault.key_vault_name
}

output "application_insights_connection_string" {
  value     = module.monitoring.application_insights_connection_string
  sensitive = true
}
