#!/usr/bin/env python
"""Test the fixed /api/query endpoint."""

import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/query"

print("=" * 60)
print("Test 1: Grounding with only image_ids (no files)")
print("=" * 60)
response = requests.post(
    BASE_URL,
    data={
        'query': 'Locate the buildings in this image',
        'image_ids': 'img_3ab2a247cbda',
    }
)
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'Task: {data["task_detected"]}')
    print(f'Route: {data["execution_summary"]["task_route"]}')
    print('✅ SUCCESS!')
else:
    print(f'❌ Error: {response.text}')

print()
print("=" * 60)
print("Test 2: Change detection with two image_ids")
print("=" * 60)
response = requests.post(
    BASE_URL,
    data={
        'query': 'What changed between these two images?',
        'image_ids': 'img_511df47fd28b,img_29a591cca944',
        'metadata': json.dumps({'dates': ['2022-05-01', '2025-05-01']})
    }
)
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'Task: {data["task_detected"]}')
    print(f'Route: {data["execution_summary"]["task_route"]}')
    print('✅ SUCCESS!')
else:
    print(f'❌ Error: {response.text}')

print()
print("=" * 60)
print("Test 3: Fusion with optical + SAR image_ids")
print("=" * 60)
response = requests.post(
    BASE_URL,
    data={
        'query': 'Does the SAR confirm the optical pattern?',
        'image_ids': 'img_60c83348bc2d,img_e36177a68b50',
        'metadata': json.dumps({'modalities': ['optical', 'sar']})
    }
)
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'Task: {data["task_detected"]}')
    print(f'Route: {data["execution_summary"]["task_route"]}')
    print('✅ SUCCESS!')
else:
    print(f'❌ Error: {response.text}')

print()
print("=" * 60)
print("All manual API tests completed!")
print("=" * 60)
