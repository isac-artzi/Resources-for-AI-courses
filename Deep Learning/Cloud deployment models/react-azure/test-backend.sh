#!/bin/bash
# Backend API Testing Script
# This script tests all API endpoints to verify the backend is working correctly

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Backend URL (change if needed)
BACKEND_URL="${1:-http://localhost:8000}"

echo -e "${YELLOW}Testing Backend API at: ${BACKEND_URL}${NC}\n"

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
response=$(curl -s -w "\n%{http_code}" "${BACKEND_URL}/")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "Response: $body"
else
    echo -e "${RED}✗ Health check failed (HTTP $http_code)${NC}"
    exit 1
fi

echo ""

# Test 2: Get all todos (should be empty initially)
echo -e "${YELLOW}Test 2: Get all todos${NC}"
response=$(curl -s -w "\n%{http_code}" "${BACKEND_URL}/api/todos")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ Get todos passed${NC}"
    echo "Response: $body"
else
    echo -e "${RED}✗ Get todos failed (HTTP $http_code)${NC}"
    exit 1
fi

echo ""

# Test 3: Create a todo
echo -e "${YELLOW}Test 3: Create a new todo${NC}"
response=$(curl -s -w "\n%{http_code}" -X POST "${BACKEND_URL}/api/todos" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Todo", "description": "This is a test", "completed": false}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "201" ]; then
    echo -e "${GREEN}✓ Create todo passed${NC}"
    echo "Response: $body"
    # Extract todo ID for next tests (assuming id is the first number in response)
    TODO_ID=$(echo "$body" | grep -o '"id":[0-9]*' | head -1 | grep -o '[0-9]*')
    echo "Created todo with ID: $TODO_ID"
else
    echo -e "${RED}✗ Create todo failed (HTTP $http_code)${NC}"
    exit 1
fi

echo ""

# Test 4: Get specific todo
if [ -n "$TODO_ID" ]; then
    echo -e "${YELLOW}Test 4: Get todo by ID${NC}"
    response=$(curl -s -w "\n%{http_code}" "${BACKEND_URL}/api/todos/${TODO_ID}")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓ Get todo by ID passed${NC}"
        echo "Response: $body"
    else
        echo -e "${RED}✗ Get todo by ID failed (HTTP $http_code)${NC}"
    fi

    echo ""

    # Test 5: Update todo
    echo -e "${YELLOW}Test 5: Update todo${NC}"
    response=$(curl -s -w "\n%{http_code}" -X PUT "${BACKEND_URL}/api/todos/${TODO_ID}" \
      -H "Content-Type: application/json" \
      -d '{"completed": true}')
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓ Update todo passed${NC}"
        echo "Response: $body"
    else
        echo -e "${RED}✗ Update todo failed (HTTP $http_code)${NC}"
    fi

    echo ""

    # Test 6: Delete todo
    echo -e "${YELLOW}Test 6: Delete todo${NC}"
    http_code=$(curl -s -w "%{http_code}" -X DELETE "${BACKEND_URL}/api/todos/${TODO_ID}")

    if [ "$http_code" = "204" ]; then
        echo -e "${GREEN}✓ Delete todo passed${NC}"
    else
        echo -e "${RED}✗ Delete todo failed (HTTP $http_code)${NC}"
    fi

    echo ""
fi

# Test 7: API Documentation
echo -e "${YELLOW}Test 7: API Documentation${NC}"
http_code=$(curl -s -w "%{http_code}" -o /dev/null "${BACKEND_URL}/docs")

if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ API documentation accessible${NC}"
    echo "Swagger UI available at: ${BACKEND_URL}/docs"
else
    echo -e "${RED}✗ API documentation not accessible (HTTP $http_code)${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All tests completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "You can view the API documentation at:"
echo "${BACKEND_URL}/docs"
