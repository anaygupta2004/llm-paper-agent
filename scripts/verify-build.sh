#!/bin/bash
set -e

echo "🔍 Verifying project build..."

# Clean up
rm -rf dist

# Install dependencies
npm install

# Build the project
npm run build

echo "✅ Build verification complete!"

# Check if key files exist
if [ ! -f "dist/public/index.html" ]; then
  echo "❌ Error: dist/public/index.html not found"
  exit 1
fi

if [ ! -f "dist/index.js" ]; then
  echo "❌ Error: dist/index.js not found"
  exit 1
fi

echo "✨ Build verification successful!"
