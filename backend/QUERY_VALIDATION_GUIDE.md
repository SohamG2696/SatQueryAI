# SatQuery AI — Query Validation & Normalization Guide

> **Version**: 1.0 · **Project**: SIH 26167 · **Layer**: Pre-Specialist Execution

---

## Overview

The **Query Validation & Normalization Layer** is the first processing step in every SatQuery AI request. It runs _before_ any specialist model (VLM, Grounding, Change Analysis, Fusion, Earth Engine) is invoked.

### Goals

1. **Reject** unrelated / out-of-domain queries (general knowledge, coding, recipes, jokes, etc.)
2. **Normalize** semantically equivalent queries into the same canonical task identifier and prompt
3. **Extract** structured parameters (grounding targets, spatial entities, index metrics) for downstream models
4. **Prevent** repetitive or over-long VLM generation via deterministic decoding settings
5. **Cache** inference results to avoid redundant specialist model executions

---

## Supported Intents

### Single-Image Intents

| Intent | canonical_task | Description |
|---|---|---|
| scene_description | scene_description | Scene captioning / image description |
| vqa | vqa | Visual Question Answering |
| grounding | grounding | Spatial entity detection & segmentation |

### Multi-Image / Specialist Intents

| Intent | canonical_task | Description |
|---|---|---|
| change_analysis | change_vqa | Bi-temporal change detection |
| optical_sar_analysis | fusion | Cross-modal optical + SAR fusion analysis |

### Geospatial Intents (Earth Engine)

| Intent | canonical_task | operation | Description |
|---|---|---|---|
| ndvi_analysis | gee | index_calculation | NDVI, NDWI, NDBI spectral index computation |
| area_analysis | gee | elevation_query | Elevation / DEM / SRTM queries |
| temporal_analysis | gee | imagery_retrieval | Sentinel / Landsat imagery retrieval |

---

## Canonical Prompts

| Task | Canonical Fixed Prompt |
|---|---|
| scene_description | Describe this satellite image in detail, including landscape features, land cover, terrain, infrastructure, and geographical elements. |
| grounding | Locate and segment the {target} in this satellite image. |
| change_analysis | Analyze and identify the physical and environmental changes between the pre-event and post-event satellite images. |
| optical_sar_analysis | Analyze the complementary optical and SAR satellite imagery jointly to characterize surface features and verify structures. |
| vqa | (User query preserved, normalized with ? suffix) |
| gee | (User raw query passed to Earth Engine planner) |

---

## Validation Decision Order

1. Empty Query -> scene_description (if image present) or invalid
2. Explicit Rejection Patterns -> invalid
3. Scene Description Paraphrases -> scene_description
4. Optical-SAR Fusion Keywords -> optical_sar_analysis
5. Bi-Temporal Change Keywords -> change_analysis
6. Geospatial / GEE Keywords -> ndvi_analysis / area_analysis / temporal_analysis
7. Spatial Grounding Prefixes -> grounding
8. VQA (Question Format or Domain Keywords) -> vqa
9. Fallback -> invalid

---

## VLM Deterministic Decoding

All VLM inference uses:
- do_sample=False (greedy, deterministic)
- max_new_tokens=100
- repetition_penalty=1.2
- no_repeat_ngram_size=3
- eos_token_id / pad_token_id resolved from tokenizer/model config

Post-processing deduplication removes consecutive duplicate sentences.

---

## Files Reference

| File | Role |
|---|---|
| app/agent/query_validator.py | Core validation & normalization logic |
| app/services/cache_service.py | LRU inference result cache (500 entries) |
| app/agent/controller.py | Wires validation layer into agent pipeline |
| app/schemas/response.py | Extended QueryResponse schema |
| models/vlm/vlm_adapter.py | Deterministic VLM generation settings |
| tests/test_query_validation_normalization.py | Comprehensive test suite (53 tests) |

---

## Test Coverage

Run: ../.venv/Scripts/python.exe -m pytest tests/test_query_validation_normalization.py -v

Expected: 53 tests passed.

| Test Group | Count |
|---|---|
| Scene description paraphrases | 12 + 1 |
| Valid VQA queries | 11 |
| Spatial grounding queries | 11 |
| Invalid / unrelated queries | 12 |
| Multi-image & geospatial intents | 3 |
| Controller invalid rejection | 1 |
| Cache consistency | 1 |
| VLM deterministic decoding | 1 |
| Total | 53 |
