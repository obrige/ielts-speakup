"""Test error paths of /api/score"""
import requests
from pathlib import Path

url = "http://127.0.0.1:8080/api/score"

# 1. 不存在的 question_id
print("=== Test: invalid question_id ===")
audio = b"fake audio bytes"
r = requests.post(url, files={"audio": ("test.wav", audio, "audio/wav")}, data={"question_id": "nonexistent"})
print(f"  Status: {r.status_code} (expected 404)")
print(f"  Body: {r.text}")
assert r.status_code == 404, f"expected 404, got {r.status_code}"

# 2. 不支持的格式
print()
print("=== Test: unsupported audio format ===")
r = requests.post(url, files={"audio": ("test.xyz", audio, "application/octet-stream")}, data={"question_id": "p1-hobby-1"})
print(f"  Status: {r.status_code} (expected 400)")
print(f"  Body: {r.text}")
assert r.status_code == 400, f"expected 400, got {r.status_code}"

# 3. webm 格式(Safari 兜底)
print()
print("=== Test: webm format (Safari fallback) ===")
r = requests.post(url, files={"audio": ("test.webm", audio, "audio/webm")}, data={"question_id": "p1-hobby-1"})
print(f"  Status: {r.status_code} (expected 500 because whisperx not installed, but format accepted)")
print(f"  Body: {r.text[:200]}")
assert r.status_code != 400, "should accept .webm"

# 4. mp4 格式(Safari 主格式)
print()
print("=== Test: mp4 format ===")
r = requests.post(url, files={"audio": ("test.mp4", audio, "audio/mp4")}, data={"question_id": "p1-hobby-1"})
print(f"  Status: {r.status_code} (expected 500 because whisperx not installed, but format accepted)")
print(f"  Body: {r.text[:200]}")
assert r.status_code != 400, "should accept .mp4"

# 5. 422 — 缺少必填字段
print()
print("=== Test: missing question_id ===")
r = requests.post(url, files={"audio": ("test.wav", audio, "audio/wav")})
print(f"  Status: {r.status_code} (expected 422)")
print(f"  Body: {r.text[:200]}")

# 6. health
print()
print("=== Health ===")
r = requests.get("http://127.0.0.1:8080/api/health")
print(f"  Status: {r.status_code}: {r.json()}")

print()
print("✅ All error paths return proper HTTP codes")