#!/bin/bash
# Usage: bash scripts/log_run.sh "Run name"
# Archives experimental runs with full metadata for reproducibility

set -e

RUN_NAME=${1:-"unnamed_run"}
DATE=$(date +"%Y%m%d_%H%M%S")
OUTDIR="run_logs/${DATE}_${RUN_NAME}"

echo "🔬 Archiving experimental run: $RUN_NAME"
echo "📁 Output directory: $OUTDIR"

# Create output directory
mkdir -p "$OUTDIR"

# Copy reports directory
if [ -d "reports/" ]; then
    echo "  ✓ Copying reports/"
    cp -r reports/ "$OUTDIR/" 2>/dev/null || true
else
    echo "  ⚠ No reports/ directory found"
fi

# Copy configs directory
if [ -d "configs/" ]; then
    echo "  ✓ Copying configs/"
    cp -r configs/ "$OUTDIR/" 2>/dev/null || true
else
    echo "  ⚠ No configs/ directory found"
fi

# Copy logs directory
if [ -d "logs/" ]; then
    echo "  ✓ Copying logs/"
    cp -r logs/ "$OUTDIR/" 2>/dev/null || true
else
    echo "  ⚠ No logs/ directory found"
fi

# Copy hw_phase5.json if it exists
if [ -f "hw_phase5.json" ]; then
    echo "  ✓ Copying hw_phase5.json"
    cp hw_phase5.json "$OUTDIR/" 2>/dev/null || true
fi

# Save git commit hash
if git rev-parse HEAD > /dev/null 2>&1; then
    echo "  ✓ Saving git commit hash"
    git rev-parse HEAD > "$OUTDIR/commit_hash.txt" 2>/dev/null || true
    git status --short > "$OUTDIR/git_status.txt" 2>/dev/null || true
else
    echo "  ⚠ Not a git repository, skipping commit hash"
fi

# Generate run metadata using Python
echo "  ✓ Generating run metadata"
python - <<'PY'
import torch
import json
import time
import os
import sys
import platform

data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "timestamp_unix": int(time.time()),
    "run_name": os.environ.get("RUN_NAME", "unnamed_run"),
    "python_version": sys.version,
    "platform": platform.platform(),
    "cuda_available": torch.cuda.is_available(),
}

if torch.cuda.is_available():
    data["gpu_name"] = torch.cuda.get_device_name(0)
    data["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    data["gpu_count"] = torch.cuda.device_count()
    data["cuda_version"] = torch.version.cuda
else:
    data["gpu_name"] = None
    data["gpu_memory_gb"] = None
    data["gpu_count"] = 0
    data["cuda_version"] = None

with open("run_metadata.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"  ✓ GPU: {data['gpu_name']}")
print(f"  ✓ CUDA: {data['cuda_available']}")
PY

# Move metadata to output directory
if [ -f "run_metadata.json" ]; then
    mv run_metadata.json "$OUTDIR/"
fi

# Create compressed tar archive
echo "  ✓ Creating compressed archive"
tar -czf "$OUTDIR.tar.gz" "$OUTDIR" 2>/dev/null

# Get archive size
ARCHIVE_SIZE=$(du -h "$OUTDIR.tar.gz" | cut -f1)

echo ""
echo "✅ Run archived successfully!"
echo "   📦 Archive: $OUTDIR.tar.gz"
echo "   💾 Size: $ARCHIVE_SIZE"
echo "   📂 Directory: $OUTDIR/"
echo ""
echo "To extract: tar -xzf $OUTDIR.tar.gz"

