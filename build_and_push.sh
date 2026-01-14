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

# Build green agent
echo "==> Building green agent..."
docker build -t ghcr.io/mpnikhil/webshop-plus-green:$VERSION \
  -f green_agent/Dockerfile .

# Build purple agent
echo "==> Building purple agent..."
docker build -t ghcr.io/mpnikhil/webshop-plus-purple:$VERSION \
  -f purple_agent/Dockerfile .

# Tag as latest if building a version
if [ "$VERSION" != "latest" ]; then
  echo "==> Tagging as latest..."
  docker tag ghcr.io/mpnikhil/webshop-plus-green:$VERSION \
    ghcr.io/mpnikhil/webshop-plus-green:latest
  docker tag ghcr.io/mpnikhil/webshop-plus-purple:$VERSION \
    ghcr.io/mpnikhil/webshop-plus-purple:latest
fi

echo "==> Build complete!"

# Push if requested
if [ "$PUSH" = true ]; then
  echo "==> Pushing to ghcr.io..."

  docker push ghcr.io/mpnikhil/webshop-plus-green:$VERSION
  docker push ghcr.io/mpnikhil/webshop-plus-purple:$VERSION

  if [ "$VERSION" != "latest" ]; then
    docker push ghcr.io/mpnikhil/webshop-plus-green:latest
    docker push ghcr.io/mpnikhil/webshop-plus-purple:latest
  fi

  echo "==> Push complete!"
fi

echo ""
echo "Images built:"
echo "  - ghcr.io/mpnikhil/webshop-plus-green:$VERSION"
echo "  - ghcr.io/mpnikhil/webshop-plus-purple:$VERSION"

if [ "$PUSH" = false ]; then
  echo ""
  echo "To push these images, run:"
  echo "  $0 --push"
fi
