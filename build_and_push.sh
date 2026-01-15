#!/bin/bash
# Build and push WebShop+ Docker images
#
# Usage:
#   ./build_and_push.sh              # Build only
#   ./build_and_push.sh --push       # Build and push
#   ./build_and_push.sh --tag v1.0   # Build with version tag

set -e

VERSION="latest"
PUSH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --push)
      PUSH=true
      shift
      ;;
    --tag)
      VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--push] [--tag VERSION]"
      exit 1
      ;;
  esac
done

echo "==> Building WebShop+ images (version: $VERSION)"

# Determine platform
PLATFORM="linux/amd64"
if [ "$PUSH" = false ]; then
  # Use host architecture for local builds to avoid slow emulation
  PLATFORM=$(docker info --format '{{.OSType}}/{{.Architecture}}')
  echo "==> Local build detected, using native platform: $PLATFORM"
else
  echo "==> Push detected, forcing platform: $PLATFORM"
fi

# Build green agent
echo "==> Building green agent..."
TAGS="-t ghcr.io/mpnikhil/webshop-plus-green:$VERSION"
if [ "$VERSION" != "latest" ]; then
  TAGS="$TAGS -t ghcr.io/mpnikhil/webshop-plus-green:latest"
fi

if [ "$PUSH" = true ]; then
  docker buildx build --platform $PLATFORM $TAGS -f green_agent/Dockerfile --push .
else
  docker buildx build --platform $PLATFORM $TAGS -f green_agent/Dockerfile --load .
fi

# Build purple agent
echo "==> Building purple agent..."
TAGS="-t ghcr.io/mpnikhil/webshop-plus-purple:$VERSION"
if [ "$VERSION" != "latest" ]; then
  TAGS="$TAGS -t ghcr.io/mpnikhil/webshop-plus-purple:latest"
fi

if [ "$PUSH" = true ]; then
  docker buildx build --platform $PLATFORM $TAGS -f purple_agent/Dockerfile --push .
else
  docker buildx build --platform $PLATFORM $TAGS -f purple_agent/Dockerfile --load .
fi

echo "==> Build and push complete!"

echo ""
echo "Images built:"
echo "  - ghcr.io/mpnikhil/webshop-plus-green:$VERSION"
echo "  - ghcr.io/mpnikhil/webshop-plus-purple:$VERSION"

if [ "$PUSH" = false ]; then
  echo ""
  echo "To push these images, run:"
  echo "  $0 --push"
fi
