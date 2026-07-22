#!/bin/bash
# Azure Deployment Commands
# This script contains all the commands needed to deploy the app to Azure
# Run each section step by step, not all at once

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Azure Deployment Script${NC}"
echo "This script contains deployment commands."
echo "Copy and run each section manually, one at a time."
echo ""

# =============================================================================
# SECTION 1: VARIABLES (Customize these!)
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 1: Set your variables (CUSTOMIZE THESE!)
# ============================================================

# Resource names (customize as needed)
export RESOURCE_GROUP="rg-todo-app"
export LOCATION="eastus"
export CONTAINER_APP_NAME="todo-api"
export CONTAINER_REGISTRY_NAME="todoappreg$(date +%s)"  # Adds timestamp for uniqueness
export ENVIRONMENT_NAME="todo-app-env"
export STATIC_WEB_APP_NAME="todo-frontend"

echo "Variables set:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Container App: $CONTAINER_APP_NAME"
echo "  Registry: $CONTAINER_REGISTRY_NAME"

EOF

# =============================================================================
# SECTION 2: LOGIN AND RESOURCE GROUP
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 2: Login to Azure and create resource group
# ============================================================

# Login to Azure
az login

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Verify resource group created
az group show --name $RESOURCE_GROUP

EOF

# =============================================================================
# SECTION 3: CONTAINER REGISTRY
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 3: Create Azure Container Registry (ACR)
# ============================================================

# Create container registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_REGISTRY_NAME \
  --sku Basic \
  --admin-enabled true

# Login to ACR
az acr login --name $CONTAINER_REGISTRY_NAME

# Get ACR login server
export ACR_LOGIN_SERVER=$(az acr show \
  --name $CONTAINER_REGISTRY_NAME \
  --query loginServer \
  --output tsv)

echo "ACR Login Server: $ACR_LOGIN_SERVER"

EOF

# =============================================================================
# SECTION 4: BUILD AND PUSH DOCKER IMAGE
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 4: Build and push Docker image
# ============================================================

# Navigate to backend directory
cd backend

# Build Docker image
docker build -t $ACR_LOGIN_SERVER/todo-api:latest .

# Push to ACR
docker push $ACR_LOGIN_SERVER/todo-api:latest

# Verify image was pushed
az acr repository list --name $CONTAINER_REGISTRY_NAME --output table

# Navigate back to root
cd ..

EOF

# =============================================================================
# SECTION 5: CONTAINER APPS ENVIRONMENT
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 5: Create Container Apps environment
# ============================================================

# Create environment
az containerapp env create \
  --name $ENVIRONMENT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Verify environment created
az containerapp env show \
  --name $ENVIRONMENT_NAME \
  --resource-group $RESOURCE_GROUP

EOF

# =============================================================================
# SECTION 6: DEPLOY CONTAINER APP
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 6: Deploy Container App
# ============================================================

# Get ACR credentials
export ACR_USERNAME=$(az acr credential show \
  --name $CONTAINER_REGISTRY_NAME \
  --query username \
  --output tsv)

export ACR_PASSWORD=$(az acr credential show \
  --name $CONTAINER_REGISTRY_NAME \
  --query passwords[0].value \
  --output tsv)

# Create container app
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $ACR_LOGIN_SERVER/todo-api:latest \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 1

# Get backend URL
export BACKEND_URL=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo ""
echo "=========================================="
echo "Backend deployed successfully!"
echo "Backend URL: https://$BACKEND_URL"
echo "=========================================="
echo ""
echo "Test it:"
echo "curl https://$BACKEND_URL/"
echo "curl https://$BACKEND_URL/api/todos"
echo ""

# Save backend URL for later
echo "VITE_API_URL=https://$BACKEND_URL" > frontend/.env.production

EOF

# =============================================================================
# SECTION 7: DEPLOY STATIC WEB APP
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 7: Deploy Static Web App (via Azure Portal recommended)
# ============================================================

echo "For Static Web App deployment, use Azure Portal:"
echo ""
echo "1. Go to https://portal.azure.com"
echo "2. Create a resource > Static Web App"
echo "3. Select your GitHub repository"
echo "4. Configure:"
echo "   - Resource Group: $RESOURCE_GROUP"
echo "   - App Location: /frontend"
echo "   - Output Location: dist"
echo "   - Build Preset: React"
echo ""
echo "5. After deployment, add GitHub secret:"
echo "   VITE_API_URL = https://$BACKEND_URL"
echo ""

# Alternative: CLI deployment (if you have GitHub token)
# az staticwebapp create \
#   --name $STATIC_WEB_APP_NAME \
#   --resource-group $RESOURCE_GROUP \
#   --source https://github.com/YOUR_USERNAME/YOUR_REPO \
#   --location "East US 2" \
#   --branch main \
#   --app-location "frontend" \
#   --output-location "dist" \
#   --login-with-github

EOF

# =============================================================================
# SECTION 8: UPDATE CORS (IF NEEDED)
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 8: Update CORS settings (if needed)
# ============================================================

# After deploying frontend, if you get CORS errors:

# 1. Get your Static Web App URL from Azure Portal
# 2. Update backend/app/main.py origins list
# 3. Rebuild and redeploy:

cd backend

# Rebuild image
docker build -t $ACR_LOGIN_SERVER/todo-api:latest .

# Push to ACR
docker push $ACR_LOGIN_SERVER/todo-api:latest

# Update container app
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_LOGIN_SERVER/todo-api:latest

cd ..

EOF

# =============================================================================
# SECTION 9: MONITORING
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 9: Monitor your deployment
# ============================================================

# View container app logs
az containerapp logs show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --tail 50 \
  --follow

# Check container app status
az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.runningStatus

EOF

# =============================================================================
# SECTION 10: CLEANUP
# =============================================================================
cat << 'EOF'

# ============================================================
# STEP 10: Cleanup (when you're done)
# ============================================================

# Delete all resources (WARNING: This deletes everything!)
# az group delete --name $RESOURCE_GROUP --yes --no-wait

# To delete just the container app:
# az containerapp delete \
#   --name $CONTAINER_APP_NAME \
#   --resource-group $RESOURCE_GROUP \
#   --yes

EOF

echo ""
echo -e "${GREEN}Script reference complete!${NC}"
echo "Copy and run each section manually."
