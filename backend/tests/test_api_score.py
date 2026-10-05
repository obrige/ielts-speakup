"""Test /api/score endpoint with real WAV file"""
import requests
from pathlib import Path
import json
import time

url = "http://127.0.0.1:8080/api/score"
audio_path = Path(__file__).parent / "test.wav"

print(f"Uploading {audio_path} ({audio_path.stat().st_size} bytes)...")
start = time.time()
with open(audio_path, "rb") as f:
    files = {"audio": (audio_path.name, f, "audio/wav")}
    data = {"question_id": "p1-hobby-1"}
    try:
        r = requests.post(url, files=files, data=data, timeout=180)
    except Exception as e:
        print(f"Network error: {e}")
        raise

elapsed = time.time() - start
print(f"Status: {r.status_code}")
print(f"Elapsed: {elapsed:.1f}s")

if r.status_code == 200:
    body = r.json()
    print()
    print("=== Response keys ===")
    for k in body.keys():
        print(f"  {k}")
    print()
    print(f"Overall: {body.get('scores', {}).get('Overall')}")
    print(f"Main scores: {body.get('scores')}")
    print(f"CEFR: {body.get('metadata', {}).get('cefr_level')}")
    print(f"Sub-dim count: {len(body.get('sub_scores', []))}")
    print(f"Main-dim count: {len(body.get('main_score_details', []))}")
    print(f"Sentence feedback: {len(body.get('sentence_feedback', []))}")
    print(f"ZPD examples: {len(body.get('zpd_examples', []))}")
    print(f"Summary keys: {list((body.get('summary') or {}).keys())}")
else:
    print(f"Body: {r.text[:500]}")