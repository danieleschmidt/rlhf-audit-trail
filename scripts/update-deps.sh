#!/bin/bash
# Script to update and compile Python dependencies
# Usage: ./scripts/update-deps.sh

set -e

echo "🔄 Updating Python dependencies..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pip-tools is installed
if ! command -v pip-compile &> /dev/null; then
    echo -e "${YELLOW}Installing pip-tools...${NC}"
    pip install pip-tools
fi

# Backup existing lock files
if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements.txt.bak
    echo -e "${GREEN}✅ Backed up requirements.txt${NC}"
fi

if [ -f "requirements-dev.txt" ]; then
    cp requirements-dev.txt requirements-dev.txt.bak
    echo -e "${GREEN}✅ Backed up requirements-dev.txt${NC}"
fi

# Compile production requirements
echo -e "${YELLOW}📦 Compiling production requirements...${NC}"
pip-compile --upgrade --resolver=backtracking --strip-extras requirements.in

# Compile development requirements
echo -e "${YELLOW}📦 Compiling development requirements...${NC}"
pip-compile --upgrade --resolver=backtracking --strip-extras requirements-dev.in

# Check for vulnerabilities in updated dependencies
echo -e "${YELLOW}🔒 Checking for security vulnerabilities...${NC}"
if command -v safety &> /dev/null; then
    safety check -r requirements.txt || echo -e "${RED}⚠️ Security vulnerabilities found in dependencies${NC}"
else
    echo -e "${YELLOW}Safety not installed, skipping vulnerability check${NC}"
fi

# Generate dependency tree
if command -v pipdeptree &> /dev/null; then
    echo -e "${YELLOW}🌳 Generating dependency tree...${NC}"
    pipdeptree --freeze > dependency-tree.txt
    echo -e "${GREEN}✅ Dependency tree saved to dependency-tree.txt${NC}"
fi

# Check for license compatibility
if command -v pip-licenses &> /dev/null; then
    echo -e "${YELLOW}📄 Checking license compatibility...${NC}"
    pip-licenses --format=json > licenses.json
    echo -e "${GREEN}✅ License information saved to licenses.json${NC}"
fi

echo -e "${GREEN}✅ Dependencies updated successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Review the changes in requirements.txt and requirements-dev.txt"
echo "2. Test the application with updated dependencies"
echo "3. Run security scans: make security"
echo "4. Commit the updated lock files"
echo ""
echo "To install updated dependencies:"
echo "  pip install -r requirements-dev.txt"