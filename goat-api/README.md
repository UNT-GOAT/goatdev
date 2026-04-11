# Goat Grading API - `goat-api`

AI-powered grading service that analyzes goat and lamb images using YOLO segmentation models and returns body measurements plus a grade. Runs as a Docker container on EC2 (port 8000).

## What It Does

- Receives 3 images (side, top, front views) + live weight from the Pi
- Runs YOLO instance segmentation on each view to extract the animal's body mask
- Measures heights (side view), widths at anatomical positions (top view), and chest width (front view)
- Converts pixel measurements to centimeters using per-camera calibration files
- Calculates a grade from measurements + weight
- Generates debug overlay images with measurement annotations
- Archives raw images and debug overlays to S3 on successful grades
- Serves debug images via `/debug/{serial_id}/{view}` for the dashboard's grade review modal

## Architecture Decisions

**CPU-only PyTorch** - the EC2 instance is a t3.medium (no GPU). A GPU instance would be faster but costs 10x more for this workload.

**API key auth (not JWT)** - Pi-to-EC2 is service-to-service. A shared API key in `X-API-Key` header is simpler than JWT for this use case. The key is in environment variables on both machines.

**S3 archival on success only** - failed grades (missing views, bad detections) are not archived. This prevents polluting the buckets with unusable data. Archival runs in a background thread so it never blocks the API response.

**Debug image cleanup** - keeps only the 100 most recent serial IDs worth of debug images on disk. Older directories are pruned after each new analysis.

**Thread-safe inference** - a single `_inference_lock` serializes all YOLO inference calls. The models are not thread-safe, and concurrent inference would OOM on a 4GB instance.

## Measurement Strategy

- **Side view**: mask divided into thirds (front/middle/rear). Extracts head height, withers height, and rump height from ground level to the top of each third.
- **Top view**: widths measured at anatomical positions (shoulder, waist, rump) defined as configurable percentages from the tail end. Goat orientation (head left/right) is configured in `config.py`.
- **Front view**: chest width from the bounding box of the largest detected contour.
- **Calibration**: each camera has a `pixels_per_cm` value from a calibration tool. Measurements are converted from pixels to centimeters using these values.

## Directory Structure

```
goat-api/
├── Dockerfile              # python:3.11-slim + OpenCV system deps, copies model/ from repo root
├── requirements.txt        # PyTorch CPU, ultralytics, OpenCV, FastAPI, boto3
├── README.md
└── api/
    ├── __init__.py         # Package exports
    ├── main.py             # App setup, lifespan (model loading, S3 config), /analyze and /health endpoints
    ├── grader.py           # YOLO model loading, inference lock, measurement extraction from masks
    ├── grade_calculator.py # Tier tables, CI/MDR computation, grade assignment, reasoning details
    ├── api_auth.py         # API key middleware (X-API-Key header)
    ├── config.py           # Model paths, S3 buckets, thresholds, goat orientation config
    ├── models.py           # Pydantic schemas (AnalyzeResponse, MeasurementsResponse, etc.)
    ├── s3.py               # Lazy S3 client, archive_in_background() thread spawning
    ├── image_validation.py # Image validation, serial_id sanitization
    ├── image_debug.py      # save_debug_images(), old directory pruning, /debug endpoints via APIRouter
    └── logger.py           # Structured logging ([LEVEL] [component] message | key=value)

model/                      # At repo root, NOT inside goat-api/
├── side/
│   ├── best.pt             # YOLO segmentation weights (side view)
│   └── side_calibration.json
├── top/
│   ├── best.pt             # YOLO segmentation weights (top view)
│   └── top_calibration.json
└── front/
    ├── best.pt             # YOLO segmentation weights (front view)
    └── front_calibration.json
```

**Important**: The Dockerfile must be built from the repository root (`docker build -f goat-api/Dockerfile .`) because it copies `model/` from outside the `goat-api/` directory.

## Environment Variables

| Variable              | Required | Description                                                     |
| --------------------- | -------- | --------------------------------------------------------------- |
| `API_KEY`             | Yes      | Shared secret for Pi authentication (must match Pi's `API_KEY`) |
| `S3_CAPTURES_BUCKET`  | Yes      | Bucket for raw capture images                                   |
| `S3_PROCESSED_BUCKET` | Yes      | Bucket for debug overlays                                       |
| `AWS_REGION`          | No       | AWS region (default: `us-east-2`)                               |

## Grade Algorithm

Grades are calculated from three inputs per animal:

**Live weight (lbs)** - entered by the operator

**Condition Index (CI)** - avg_body_width_cm / withers_height_cm - measures overall body fill. A wider animal relative to its height indicates better condition.

**Muscle Distribution Ratio (MDR)** - rump_width_cm / waist_width_cm - measures hindquarter muscle development. A higher ratio means more muscle mass in the rear, which correlates with meat yield.

Grade tiers are defined per category (species × description: meat/dairy/cross for goats, lamb/ewe for lambs). An animal must meet all three minimums (weight, CI, MDR) to qualify for a tier. Tiers are checked top-down from best to worst - first match wins. Animals below all tiers receive "No Roll."
Graceful degradation: if a view fails and a ratio can't be computed, the algorithm grades with whatever data is available rather than returning no grade at all. Missing ratios are treated as passing, so the grade is effectively weight + the available ratio.
These thresholds are facility-specific live intake tiers calibrated from facility operator knowledge, initial measurements, and broad market sanity checks. They are not yet intended to represent official USDA species-wide grade nomenclature.

They will be refined as more animals are graded and validated against manual grades from the facility.

## Facility Grading Specification

Facility-specific live intake classes and grade tiers for goat and lamb sorting at Clean Chickens and Co.

These labels are facility-specific live-animal intake classes used for internal sorting. They are not yet intended to represent official USDA species-wide grade nomenclature.

### Purpose

This document defines the current grading structure used by the HerdSync grading system. It is designed to match the facility's existing live-animal workflow as closely as possible while making the logic explicit enough to document, audit, and improve over time.

The grading system is intentionally restrictive. It relies on:

- Operator-entered live weight in pounds
- Facility category assignment
- Two visual conformation ratios derived from camera measurements

The system is not trying to reproduce a national USDA live grading program for goats or lambs. It is trying to produce a consistent, documented version of the facility's current intake and pricing workflow.

### Scope

This specification applies to live animals graded in HerdSync for:

- Goat: Meat
- Goat: Cross
- Goat: Dairy
- Lamb: Lamb
- Lamb: Ewe

These are facility categories, not claims of official industry taxonomy.

### Data Inputs

Every grade is derived from three inputs:

1. `Live Weight (lbs)`
2. `Condition Index (CI) = avg_body_width_cm / withers_height_cm`
3. `Muscle Distribution Ratio (MDR) = rump_width_cm / waist_width_cm`

#### Condition Index

CI is a width-to-height proxy for overall body fill. Higher values indicate a relatively wider, deeper animal for its height.

Operational interpretation:

- Higher CI generally supports thicker, fuller animals
- Lower CI generally supports narrower, lighter-conditioned animals
- CI is used as a conformation screen, not as an official USDA trait

#### Muscle Distribution Ratio

MDR is a rump-to-waist width proxy for hindquarter development. Higher values indicate more rear-end shape relative to the waist.

Operational interpretation:

- Higher MDR generally supports stronger hindquarter expression
- Lower MDR generally supports flatter or less muscular rear shape
- MDR is used as an internal meat-yield proxy, not as an official USDA trait

### Grade Assignment Rule

Each category has a fixed threshold table. An animal must meet all listed minimums for a tier:

- minimum live weight
- minimum CI
- minimum MDR

Tiers are evaluated from highest to lowest. The first matching tier is returned.

Animals that do not meet the lowest tier in their category receive `No Roll`.

#### Partial Measurement Fallback

If one ratio cannot be computed because a required measurement is missing, HerdSync still attempts a grade using:

- weight
- the ratio that is available

The missing ratio is treated as passing for that calculation.

This fallback exists to preserve workflow continuity on imperfect image sets, but it should be treated as lower-confidence than a full three-input grade.

### Current Tier Tables

#### Goat: Meat

| Tier       | Min Weight (lbs) | Min CI | Min MDR |
| ---------- | ---------------: | -----: | ------: |
| CAB Prime  |              105 |   0.60 |    1.06 |
| Prime      |               90 |   0.55 |    1.02 |
| CAB Choice |               82 |   0.52 |    1.00 |
| Choice     |               75 |   0.48 |    0.98 |
| Select     |               45 |   0.40 |    0.90 |

#### Goat: Cross

| Tier      | Min Weight (lbs) | Min CI | Min MDR |
| --------- | ---------------: | -----: | ------: |
| CAB Prime |              100 |   0.58 |    1.04 |
| Prime     |               85 |   0.53 |    1.00 |
| Choice    |               70 |   0.46 |    0.96 |
| Select    |               45 |   0.40 |    0.90 |

#### Goat: Dairy

| Tier   | Min Weight (lbs) | Min CI | Min MDR |
| ------ | ---------------: | -----: | ------: |
| Prime  |              110 |   0.50 |    1.00 |
| Choice |               85 |   0.45 |    0.96 |
| Select |               50 |   0.40 |    0.92 |

#### Lamb: Lamb

| Tier      | Min Weight (lbs) | Min CI | Min MDR |
| --------- | ---------------: | -----: | ------: |
| CAB Prime |              125 |   0.62 |    1.10 |
| Prime     |              110 |   0.58 |    1.05 |
| Choice    |               95 |   0.52 |    1.00 |
| Select    |               80 |   0.45 |    0.95 |

#### Lamb: Ewe

| Tier   | Min Weight (lbs) | Min CI | Min MDR |
| ------ | ---------------: | -----: | ------: |
| Prime  |              150 |   0.55 |    1.02 |
| Choice |              130 |   0.50 |    0.98 |
| Select |              100 |   0.44 |    0.94 |

### Category Definitions

The category field is a facility intake bucket. It should be assigned based on how the facility already sorts and prices animals in practice.

#### Goat: Meat

Goats visually and operationally handled as meat-type animals in the facility workflow.

Expected tendencies:

- broader body shape
- stronger fill through the middle
- more hindquarter shape than dairy-type animals

#### Goat: Cross

Goats that present as mixed or intermediate type in the facility workflow and are not best represented by the facility's meat-only or dairy-only buckets.

Expected tendencies:

- mixed frame and body style
- intermediate width and shape
- inconsistent expression relative to pure meat- or dairy-type sorting

This is a house category. It should be treated as a facility sorting bucket, not an official USDA class.

#### Goat: Dairy

Goats visually and operationally handled as dairy-type animals in the facility workflow.

Expected tendencies:

- taller, narrower frame
- less body fill at equal weight
- less hindquarter expression than meat-type goats

Because dairy-type animals often appear narrower at the same live weight, this category uses a heavier weight floor than goat meat at comparable tiers.

#### Lamb: Lamb

Sheep the facility handles and prices as market lambs rather than mature ewe animals.

Expected tendencies:

- younger market-animal appearance
- stronger finish and carcass-style shape at moderate to heavy lamb weights

This is a facility bucket aligned with market handling, not a complete formal sheep taxonomy.

#### Lamb: Ewe

Sheep the facility handles and prices as ewe animals rather than market lambs.

Expected tendencies:

- mature female phenotype
- heavier mature body size
- lower expected tier at lighter live weights relative to market lambs

### Why The Structure Looks This Way

The current structure reflects the facility workflow first and formal livestock taxonomy second.

Reasons for the current design:

- The client wants continuity with current in-house language and sorting
- Operators already think in terms of category plus top-down tiering
- Live weight is available and trusted by the facility
- The camera system can reliably estimate a small number of width and height measurements
- CI and MDR are simple enough to explain and audit

This creates a practical grading system that is easier to defend operationally than an opaque model score.

### Sanity Check Summary

The current tables were checked against USDA terminology and broad extension or market references, with emphasis on whether the thresholds were directionally plausible for live animals at a Minnesota facility.

#### Key Findings

- The categories should be understood as facility labels rather than official USDA classes
- Goat `meat`, `dairy`, and especially `cross` do not function as a single official live-grade taxonomy in the same way this table presents them
- Sheep `lamb` and `ewe` are not parallel official dimensions, but they are still usable as facility buckets if that matches the workflow
- `CAB Prime` is a facility tier label in this context and should be treated that way in the documentation

#### Weight Threshold Read

- `Lamb: Ewe` is the most externally plausible set and aligns reasonably well with reported ewe weight ranges in recent graded sale reports
- `Lamb: Lamb` is somewhat heavy, but still plausible for a facility that prefers bigger, better-finished lambs
- `Goat: Dairy` is directionally sensible because dairy-type goats can require more weight to look comparably full
- `Goat: Meat` and `Goat: Cross` top tiers are stricter and more facility-specific than broad market-goat references

#### Bottom-Line Interpretation

The tables are acceptable as a facility-specific intake framework if they are documented honestly as internal sorting logic rather than official livestock grading standards.

### Research Notes

The following references informed the sanity check:

- USDA goat terminology and market descriptors: [Livestock, Poultry and Grain Goat Terms](https://www.ams.usda.gov/market-news/livestock-poultry-and-grain-goat-terms)
- USDA sheep standards and class definitions: [Slaughter Lambs, Yearlings, and Sheep Grades and Standards](https://www.ams.usda.gov/sites/default/files/media/Slaughter_Lambs%2C_Yearlings%2C_and_Sheep%5B1%5D.pdf)
- USDA Livestock, Poultry, and Grain Market News handbook: [LPGMN Reporter's Handbook](https://www.ams.usda.gov/sites/default/files/media/LPGMNReporterHandbook.pdf)
- USDA graded sheep and goat sale references used for directional weight sanity checks: [AMS_2215](https://www.ams.usda.gov/mnreports/ams_2215.pdf), [AMS graded sale example](https://mymarketnews.ams.usda.gov/filerepo/sites/default/files/3622/2026-02-09/1303194/ams_3622_00019.pdf)
- University of Minnesota consumer guidance on live animal purchase sizes: [Buying animals for meat processing](https://extension.umn.edu/save-money-food/buying-animals-meat-processing)
- Penn State guidance on ideal market goat size ranges: [Ideal market goat](https://extension.psu.edu/courses/meat-goat/basic-production/selecting-meat-goats/ideal-market-goat)

These sources were used for broad plausibility checks only. The facility thresholds remain internal business rules.

### Known Limitations

- Category assignment still depends on human input
- The model does not directly score fat cover, breed makeup, sex condition, or maturity beyond what is visually implied
- Missing measurements can elevate uncertainty because fallback grading skips unavailable ratios
- Thresholds are still subject to revision as more manually reviewed grades are collected
