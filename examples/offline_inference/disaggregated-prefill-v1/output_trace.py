import json
import gzip

# 读取 trace 文件
with gzip.open("./vllm_profile/xxx.pt.trace.json.gz", "rt") as f:
    trace_data = json.load(f)

# 解析 Forward 等事件
for event in trace_data.get("traceEvents", []):
    if event.get("name") in ["Forward", "Preprocess", "Sample", "Postprocess"]:
        print(f"Event: {event['name']}")
        print(f"  Duration: {event.get('dur', 0)} us")  # 微秒
        print(f"  Timestamp: {event.get('ts', 0)} us")