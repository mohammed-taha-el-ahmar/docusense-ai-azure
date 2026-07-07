# Remote state — configure at init:
#
#   terraform init \
#     -backend-config="resource_group_name=rg-tfstate" \
#     -backend-config="storage_account_name=sttfstatedocusense" \
#     -backend-config="container_name=tfstate" \
#     -backend-config="key=docusense-${env}.tfstate"

terraform {
  backend "azurerm" {}
}
