#!/bin/bash
set -e

BASE_URL="http://127.0.0.1:5305"

echo "1️⃣ Creating collection..."
curl -s -X POST "$BASE_URL/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_vectors",
    "dim": 3,
    "metric": "cosine",
    "hnsw": { "m": 16, "ef_construction": 200, "ef_search": 50 }
  }'
echo -e "\n✅ Collection created."

echo "2️⃣ Upserting vectors..."
curl -s -X POST "$BASE_URL/collections/my_vectors/upsert" \
  -H "Content-Type: application/json" \
  -d '{
    "ids": [1, 2, 3],
    "vectors": [
      [0.1, 0.2, 0.3],
      [0.2, 0.1, 0.9],
      [0.9, 0.1, 0.3]
    ],
    "payloads": [
      {"name": "vec1"},
      {"name": "vec2"},
      {"name": "vec3"}
    ]
  }'
echo -e "\n✅ Vectors inserted."

echo "3️⃣ Searching..."
curl -s -X POST "$BASE_URL/collections/my_vectors/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.25],
    "top_k": 2
  }'
echo -e "\n✅ Search done."

echo "4️⃣ Listing collections..."
curl -s "$BASE_URL/collections"
echo -e "\n✅ Done."
