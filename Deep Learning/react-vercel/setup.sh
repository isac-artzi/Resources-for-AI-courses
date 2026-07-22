#!/bin/bash

###############################################################################
# Automated Setup Script
# ======================
#
# This script automates the setup process for the Deep Learning application.
# It sets up both backend and frontend with all dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Author: Cloud ML Deployment
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

###############################################################################
# Main Setup
###############################################################################

print_header "Deep Learning App Setup"
echo ""

# Check prerequisites
print_header "Checking Prerequisites"
echo ""

if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python installed: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

if command_exists node; then
    NODE_VERSION=$(node --version)
    print_success "Node.js installed: $NODE_VERSION"
else
    print_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi

if command_exists npm; then
    NPM_VERSION=$(npm --version)
    print_success "npm installed: $NPM_VERSION"
else
    print_error "npm not found. Please install npm"
    exit 1
fi

echo ""

###############################################################################
# Backend Setup
###############################################################################

print_header "Setting Up Backend"
echo ""

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_info "Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
print_success "Python dependencies installed"

# Create .env file if not exists
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cp .env.example .env 2>/dev/null || echo "# Backend environment variables" > .env
    print_success ".env file created"
fi

# Test backend
print_info "Testing backend imports..."
python -c "from app.main import app; print('Backend OK')" && print_success "Backend imports successful"

cd ..

echo ""

###############################################################################
# Frontend Setup
###############################################################################

print_header "Setting Up Frontend"
echo ""

cd frontend

# Install dependencies
print_info "Installing Node.js dependencies (this may take a minute)..."
npm install
print_success "Node.js dependencies installed"

# Create .env.local file
if [ ! -f ".env.local" ]; then
    print_info "Creating .env.local file..."
    echo "VITE_API_URL=http://localhost:8000" > .env.local
    print_success ".env.local file created"
fi

# Test build
print_info "Testing frontend build..."
npm run build > /dev/null 2>&1 && print_success "Frontend build successful"

cd ..

echo ""

###############################################################################
# Summary
###############################################################################

print_header "Setup Complete!"
echo ""

echo -e "${GREEN}✓ Backend is ready${NC}"
echo -e "  Location: ./backend"
echo -e "  Virtual env: ./backend/venv"
echo ""

echo -e "${GREEN}✓ Frontend is ready${NC}"
echo -e "  Location: ./frontend"
echo -e "  Dependencies: ./frontend/node_modules"
echo ""

print_header "Next Steps"
echo ""

echo "To start the application:"
echo ""
echo -e "${YELLOW}Terminal 1 (Backend):${NC}"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload"
echo ""
echo -e "${YELLOW}Terminal 2 (Frontend):${NC}"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo -e "${YELLOW}Then open:${NC} http://localhost:5173"
echo ""

print_header "Documentation"
echo ""
echo "  📖 Full Tutorial:    README.md"
echo "  🚀 Quick Start:      QUICKSTART.md"
echo "  🧪 Testing Guide:    TESTING.md"
echo "  🌐 Deployment:       DEPLOYMENT.md"
echo ""

print_success "Setup script completed successfully!"
echo ""
