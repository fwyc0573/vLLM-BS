#!/bin/bash
# Test script to verify vLLM import

OUTPUT_FILE="/research/d1/gds/ytyang/yichengfeng/frontier/sota-infer-engine/vllm/tests/import_result.txt"

# Activate conda environment
source /research/d1/gds/ytyang/anaconda3/etc/profile.d/conda.sh
conda activate vllm-bs-0.10.2

{
    echo "=== Python and vLLM Info ==="
    echo "Date: $(date)"
    echo "Python: $(which python)"
    echo ""
    
    echo "=== pip show vllm ==="
    pip show vllm 2>&1 | head -10
    echo ""
    
    echo "=== Test vLLM Import ==="
    python -c "
import sys
print('Python:', sys.executable)
try:
    from vllm import LLM, SamplingParams
    print('SUCCESS: vllm imported!')
    import vllm
    print('vLLM path:', vllm.__file__)
    print('vLLM version:', vllm.__version__)
except Exception as e:
    print('FAILED:', e)
    import traceback
    traceback.print_exc()
" 2>&1
    
    echo ""
    echo "=== Done ==="
} > "$OUTPUT_FILE" 2>&1

echo "Output written to $OUTPUT_FILE"
cat "$OUTPUT_FILE"

