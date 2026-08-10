# B Tourism Data Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce auditable, model-ready data tables for B题 while preserving every raw source and its provenance.

**Architecture:** Read existing raw files plus the approved friend archive as immutable inputs. Normalize each source family to typed CSV outputs in `data/clean/`, attach provenance and quality flags, then create a machine-readable quality report. No inferred values are added during cleaning.

**Tech Stack:** Python 3.12, pandas, openpyxl, pytest.

---

### Task 1: Define and test core normalizers

**Files:**
- Create: `src/data/cleaning.py`
- Create: `tests/test_cleaning.py`

- [ ] Write failing tests for search-heat normalization, POI coordinate validation, and quality reporting.
- [ ] Run `pytest tests/test_cleaning.py -v` and confirm expected import failure.
- [ ] Implement the smallest normalizers to satisfy the tests.
- [ ] Re-run `pytest tests/test_cleaning.py -v`.

### Task 2: Build the repeatable cleaning pipeline

**Files:**
- Create: `scripts/clean_b_data.py`
- Modify: `src/data/cleaning.py`
- Modify: `tests/test_cleaning.py`

- [ ] Add a failing test for producing a quality report from a cleaned source table.
- [ ] Implement archive readers and source-family cleaners without modifying raw inputs.
- [ ] Write normalized tables to `data/clean/` and a `quality_report.json`.
- [ ] Run focused tests and then the full test suite.

### Task 3: Run, inspect, and document the output

**Files:**
- Create: `data/clean/*.csv`
- Create: `data/clean/quality_report.json`
- Create: `data/clean/README.md`

- [ ] Run the cleaner on the current raw data and friend archive.
- [ ] Verify schema, key uniqueness, date ranges, coordinate ranges, and source row counts.
- [ ] Record unresolved source limitations in the clean-data README.
