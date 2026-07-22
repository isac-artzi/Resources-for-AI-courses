# Azure Deployment Checklist

Use this checklist to ensure smooth deployment to Azure.

## ☑️ Pre-Deployment

- [ ] App works perfectly locally (both frontend and backend)
- [ ] All code is committed to Git
- [ ] GitHub repository is created and code is pushed
- [ ] Azure account is created ([azure.com/free](https://azure.microsoft.com/free/))
- [ ] Azure CLI is installed and logged in (`az login`)
- [ ] Docker is installed (for backend deployment)

## ☑️ Backend Deployment (Azure Container Apps)

### Resource Group Setup
- [ ] Resource group created
  ```bash
  az group create --name rg-todo-app --location eastus
  ```

### Container Registry
- [ ] Azure Container Registry (ACR) created
- [ ] Registry name is globally unique (lowercase, no special chars)
- [ ] Logged into ACR (`az acr login --name YOUR_REGISTRY`)

### Docker Image
- [ ] Docker image builds successfully locally
  ```bash
  cd backend
  docker build -t todo-api .
  ```
- [ ] Docker image tagged with ACR server
- [ ] Image pushed to ACR successfully

### Container App
- [ ] Container Apps environment created
- [ ] Container app created with correct settings:
  - [ ] External ingress enabled
  - [ ] Target port: 8000
  - [ ] Min replicas: 0 (to stay in free tier)
  - [ ] Max replicas: 1
- [ ] Backend URL noted and tested (https://your-app.azurecontainerapps.io)

### Backend Testing
- [ ] Health check endpoint works: `curl https://your-backend-url/`
- [ ] API endpoints work: `curl https://your-backend-url/api/todos`
- [ ] API documentation accessible: `https://your-backend-url/docs`

## ☑️ Frontend Deployment (Azure Static Web Apps)

### Configuration
- [ ] `frontend/.env.production` updated with backend URL
  ```env
  VITE_API_URL=https://your-backend-url.azurecontainerapps.io
  ```
- [ ] `staticwebapp.config.json` is in frontend directory
- [ ] Backend CORS settings include Static Web App URL

### Static Web App Creation
- [ ] Static Web App created via Azure Portal or CLI
- [ ] GitHub integration configured
- [ ] Build settings:
  - [ ] App location: `/frontend`
  - [ ] Output location: `dist`
  - [ ] Build preset: React

### GitHub Actions
- [ ] GitHub Action workflow file created (auto-generated)
- [ ] Workflow runs successfully
- [ ] Build completes without errors
- [ ] Deployment succeeds

### Frontend Testing
- [ ] Static Web App URL accessible
- [ ] UI loads correctly
- [ ] No console errors
- [ ] Can create todos
- [ ] Can mark todos complete
- [ ] Can delete todos
- [ ] API calls successful (check Network tab)

## ☑️ CORS Configuration

- [ ] Backend CORS origins include:
  - [ ] `http://localhost:5173` (local dev)
  - [ ] `https://your-app.azurestaticapps.net` (production)
- [ ] No CORS errors in browser console
- [ ] If CORS errors occur:
  - [ ] Updated `backend/app/main.py` with correct origins
  - [ ] Rebuilt Docker image
  - [ ] Redeployed to Container Apps

## ☑️ Cost Management

- [ ] Billing alerts set up in Azure Portal
- [ ] Resources configured for free tier:
  - [ ] Container Apps: min replicas = 0
  - [ ] Static Web Apps: Free tier selected
- [ ] Understand Container Registry costs (~$5/month)
- [ ] Know how to delete resources when done:
  ```bash
  az group delete --name rg-todo-app --yes
  ```

## ☑️ Documentation

- [ ] README.md updated with:
  - [ ] Your backend URL
  - [ ] Your frontend URL
  - [ ] Any customizations made
- [ ] Environment variables documented
- [ ] Deployment notes added

## ☑️ Security

- [ ] No secrets committed to Git (.env files in .gitignore)
- [ ] Container running as non-root user
- [ ] HTTPS enabled (automatic with Azure)
- [ ] CORS properly configured (not `allow_origins=["*"]`)

## ☑️ Post-Deployment Testing

### Functionality Tests
- [ ] Create multiple todos
- [ ] Edit todos
- [ ] Delete todos
- [ ] Filter by all/active/completed
- [ ] Test on mobile device
- [ ] Test on different browsers

### Performance Tests
- [ ] App loads in under 3 seconds
- [ ] API responses are fast (< 500ms)
- [ ] No memory leaks (check for long time)

### Error Handling
- [ ] Test with backend stopped (should show error message)
- [ ] Test with invalid data
- [ ] Check error messages are user-friendly

## ☑️ Monitoring

- [ ] Azure Portal shows "Healthy" status for all resources
- [ ] GitHub Actions showing successful builds
- [ ] Set up Application Insights (optional but recommended)
- [ ] Monitor Container Apps logs:
  ```bash
  az containerapp logs show \
    --name todo-api \
    --resource-group rg-todo-app \
    --tail 50 \
    --follow
  ```

## 🎉 Deployment Complete!

Once all items are checked:

1. **Share your app**: Give friends the Static Web App URL
2. **Monitor costs**: Check Azure Portal regularly
3. **Keep learning**: Try adding features from TUTORIAL.md
4. **Clean up**: When done, delete resources to avoid charges

---

## Troubleshooting Reference

| Issue | Solution |
|-------|----------|
| CORS error | Update backend origins, rebuild, redeploy |
| 404 on API calls | Check VITE_API_URL in .env.production |
| Backend won't start | Check logs with `az containerapp logs` |
| Build fails | Check GitHub Actions logs |
| High costs | Ensure min replicas = 0, delete unused resources |

---

**Need help?** See TUTORIAL.md troubleshooting section or check Azure Portal logs.
