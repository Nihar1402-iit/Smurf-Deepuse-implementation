#!/bin/bash
# Push to GitHub Script
# This script pushes the SMURF Ultrasound code to GitHub

set -e

echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  Pushing to GitHub: Smurf-Deepuse-implementation           ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""

REPO_URL="https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation.git"
BRANCH="main"

echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git not initialized. Run from project root."
    exit 1
fi

# Check git status
echo "Current git status:"
git status
echo ""

# Add remote if not exists
if ! git remote -v | grep -q "$REPO_URL"; then
    echo "Adding remote origin..."
    git remote add origin "$REPO_URL"
    echo "✓ Remote added"
else
    echo "✓ Remote already exists"
fi

echo ""

# Create main branch if needed
if ! git rev-parse --verify main 2>/dev/null; then
    echo "Creating main branch..."
    git branch -M main
    echo "✓ Main branch created"
fi

echo ""

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main --force-with-lease

echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  Push Complete!                                             ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""
echo "Repository URL: $REPO_URL"
echo ""
echo "Next steps:"
echo "1. Verify on GitHub: https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation"
echo "2. Clone on server: git clone $REPO_URL"
echo "3. Install: bash install.sh cu118 (for CUDA 11.8)"
echo "4. Train: python3 train_gpu.py --gpu 0 --batch-size 16"
echo "5. Test: python3 test_gpu.py --gpu 0"
echo ""
