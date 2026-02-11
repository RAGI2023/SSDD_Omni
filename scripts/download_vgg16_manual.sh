#!/bin/bash
# Manual download script for VGG16 when automatic download fails

echo "Downloading VGG16 manually..."
echo "Target: ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
echo ""

# Create directory if not exists
mkdir -p ~/.cache/torch/hub/checkpoints/

# Method 1: Try direct download with wget
echo "Method 1: Trying wget..."
wget -c https://download.pytorch.org/models/vgg16-397923af.pth \
    -O ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth.tmp

if [ $? -eq 0 ]; then
    mv ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth.tmp \
       ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
    echo "✓ Download successful!"
    echo ""
    echo "File size:"
    ls -lh ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
    exit 0
fi

# Method 2: Try with curl
echo ""
echo "Method 2: Trying curl..."
curl -L https://download.pytorch.org/models/vgg16-397923af.pth \
    -o ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth.tmp \
    -C -

if [ $? -eq 0 ]; then
    mv ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth.tmp \
       ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
    echo "✓ Download successful!"
    echo ""
    echo "File size:"
    ls -lh ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
    exit 0
fi

echo ""
echo "✗ Both methods failed. Please check your network connection."
echo ""
echo "Alternative: Download from a mirror or use a VPN, then copy to:"
echo "  ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
echo ""
echo "File info:"
echo "  URL: https://download.pytorch.org/models/vgg16-397923af.pth"
echo "  Size: ~528MB"
echo "  Hash: 397923af (first 8 chars)"
