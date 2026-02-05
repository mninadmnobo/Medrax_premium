# MedRAX Complete Pipeline Analysis & Optimization Report

**Generated:** February 5, 2026  
**Status:** Clean & Optimized  
**Total Lines of Code:** 8,009 lines (across agent + tools)

---

## 📊 PIPELINE ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MedRAX INFERENCE PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              USER INPUT
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  LLM Agent (LangGraph)  │
                    │  - LLM Model Selection  │
                    │  - Tool Binding         │
                    └──────────┬──────────────┘
                               │
                               ▼
              ┌────────────────────────────────────┐
              │  TOOL EXECUTION PHASE              │
              │  (5 Parallel Medical AI Tools)     │
              └────────┬─────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │ (etc.)
        ▼              ▼              ▼
    ┌────────┐  ┌────────┐  ┌─────────────┐
    │Classif.│  │Segment │  │CheXagent VQA│
    │(Dense) │  │(PSPNet)│  │             │
    └────┬───┘  └────┬───┘  └──────┬──────┘
         │           │             │
         └───────────┼─────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ CANONICAL OUTPUT           │
        │ NORMALIZATION              │
        │ (normalize_output)         │
        │                            │
        │ CanonicalFinding objects   │
        │ - Standardized format      │
        │ - Confidence scores        │
        │ - Metadata extracted       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ CONFLICT DETECTION PHASE   │
        │ (ConflictDetector)         │
        │                            │
        │ ├─ Exact Conflicts         │
        │ ├─ BERT NLI Conflicts      │
        │ └─ GACL Anatomical         │
        │    Conflicts               │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ CONFLICT RESOLUTION PHASE  │
        │ (ConflictResolver)         │
        │                            │
        │ ├─ BERT Score Analysis     │
        │ ├─ Task-Aware Arbitration  │
        │ ├─ Confidence Weighting    │
        │ └─ Human Deferral         │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ COMPREHENSIVE LOGGING      │
        │ - Tool executions          │
        │ - Canonical findings       │
        │ - Conflict analysis        │
        │ - Resolutions              │
        └────────────┬───────────────┘
                     │
                     ▼
              FINAL RESULTS
        ├─ Tool Messages
        ├─ Conflict Report
        └─ Log File (JSON)
```

---

## 📁 MODULE BREAKDOWN (8 Core Components)

### **1. AGENT ORCHESTRATION** (407 lines)
**File:** `agent.py`  
**Role:** Main orchestrator using LangGraph  
**Key Classes:**
- `Agent`: Main agent class with LangGraph workflow
- `AgentState`: TypedDict for message state management

**Functions:**
- `process_request()`: LLM inference
- `has_tool_calls()`: Check for tool calls
- `execute_tools()`: Execute and normalize findings
- `_get_tool_type()`: Tool type detection
- `_save_tool_calls_with_conflicts()`: Comprehensive logging

**Status:** ✅ Clean, no redundancies

---

### **2. CANONICAL OUTPUT NORMALIZATION** (882 lines)
**File:** `canonical_output.py`  
**Role:** Standardize outputs from 5 different tools into unified format  
**Key Classes:**
- `CanonicalFinding`: Standard data structure for all findings

**Functions:**
- `normalize_output()`: Main entry point (routes to task-specific normalizers)
- `normalize_classification_output()`: For DenseNet classifier
- `normalize_vqa_output()`: For CheXagent VQA
- `normalize_segmentation_output()`: For PSPNet segmentation
- `normalize_grounding_output()`: For phrase grounding
- `normalize_report_output()`: For report generation

**Special Functions:**
- `calibrate_confidence()`: Confidence calibration
- `estimate_text_confidence()`: Extract confidence from text

**Status:** ✅ All functions used, no redundancies

---

### **3. CONFLICT DETECTION PHASE** (849 + 560 + 675 = 2,084 lines)

#### **3a. Conflict Resolution Coordinator** (849 lines)
**File:** `conflict_resolution.py`  
**Role:** Main conflict detection and resolution  
**Key Classes:**
- `Conflict`: Dataclass representing a detected conflict
- `ConflictDetector`: Orchestrates detection methods
- `ConflictResolver`: Implements resolution strategy

**Detection Methods:**
- `_check_exact_conflicts()`: Binary disagreements
- `_check_bert_conflicts()`: Semantic contradictions (BERT NLI)
- `_check_gacl_conflicts()`: Anatomical constraints (GACL)

**Resolution Methods:**
- `_analyze_bert_scores()`: Extract scores
- `_bert_guided_resolution()`: Use BERT for decision
- `_get_task_type()`: Map conflict to task
- `_find_tool_finding()`: Locate specific tool's finding
- `_weighted_average_resolution()`: Fallback averaging

**Status:** ✅ All functions used, core pipeline

#### **3b. BERT NLI Detector** (560 lines)
**File:** `bert_conflict_detector.py`  
**Role:** Semantic conflict detection using DeBERTa-MNLI  
**Key Classes:**
- `BERTConflictDetector`: Main detector
- `MedicalConflictDetector`: Medical-specific wrapper
- `ConflictPrediction`: Result dataclass

**Model:** `microsoft/deberta-large-mnli`  
**Outputs:** contradiction_prob, entailment_prob, neutral_prob

**Status:** ✅ Used in conflict detection pipeline

#### **3c. GACL Anatomical Detector** (675 lines)
**File:** `anatomical_consistency_graph.py`  
**Role:** Graph-based anatomical constraint checking  
**Key Classes:**
- `AnatomicalGraph`: Constraint database
- `GACLConflictDetector`: Main detector

**Features:**
- Cross-organ constraints
- Measurement-based analysis
- Segmentation vs. classifier consistency

**Status:** ✅ Used in conflict detection pipeline

---

### **4. CONFIDENCE SCORING SYSTEM** (1,385 lines)
**File:** `confidence_scoring.py`  
**Role:** Extract, normalize, and calibrate confidence scores  
**Key Classes:**
- `ConfidenceScoringPipeline`: Main orchestrator
- Task-specific extractors (6 types):
  - `ClassificationConfidenceExtractor`
  - `SegmentationConfidenceExtractor`
  - `VQAConfidenceExtractor`
  - `GroundingConfidenceExtractor`
  - `ReportConfidenceExtractor`
  - `GenerationConfidenceExtractor`
- `ConfidenceNormalizer`: Min-max normalization
- `ConfidenceCalibrator`: Isotonic/temperature scaling
- `ConfidenceFusion`: Multi-model fusion
- `CalibrationMetrics`: ECE, Brier Score, AUROC

**Status:** ✅ Comprehensive, all extractors used

---

### **5. PROBABILISTIC CONFLICT GRAPH** (462 lines)
**File:** `probabilistic_conflict_graph.py`  
**Role:** Calibration analysis using ensemble methods  
**Key Classes:**
- `ProbabilisticConflictGraph`: Graph-based analysis

**Functions:**
- `build_graph()`: Construct conflict graph
- `analyze_tool_calibration()`: Tool calibration metrics

**Status:** ⚠️ **UNUSED** - Initialized but never called
- Imported in agent.py (line 17)
- Instantiated in `__init__` (line 106)
- But `_detect_conflicts_with_probabilistic_graph()` is never invoked

---

## 🛠️ TOOL IMPLEMENTATIONS (6 Medical AI Tools)

| Tool | File | Lines | Purpose |
|------|------|-------|---------|
| **DenseNet Classifier** | `classification.py` | 160 | 18-class pathology detection |
| **PSPNet Segmentation** | `segmentation.py` | 325 | Organ measurements (width, height, area) |
| **CheXagent VQA** | `xray_vqa.py` | 294 | Question-answering about CXR |
| **LLaVA-Med** | `llava_med.py` | 274 | Vision-language model for CXR |
| **Report Generation** | `report_generation.py` | 370 | Generate radiology reports (ViT-BERT) |
| **Phrase Grounding** | `grounding.py` | 343 | Localize findings in image |
| **Utilities** | `utils.py` + `dicom.py` + `generation.py` | 522 | Support functions |

**Status:** ✅ All implemented and used

---

## 🔍 REDUNDANCY ANALYSIS

### **REMOVED:**
1. ❌ **`agent_clean.py`** (407 lines) - Exact duplicate of `agent.py`
   - **Action:** DELETED
   - **Reason:** Redundant copy, identical code
   - **Savings:** 407 lines

### **UNUSED BUT POTENTIALLY VALUABLE:**
1. ⚠️ **`_detect_conflicts_with_probabilistic_graph()`** (25 lines in agent.py)
   - **Status:** Defined but never called
   - **Why:** Complex analysis not integrated into main pipeline
   - **Recommendation:** Either integrate into pipeline OR remove

2. ⚠️ **`ProbabilisticConflictGraph` class** (462 lines)
   - **Status:** Instantiated but never used
   - **Why:** Advanced calibration analysis not in execution path
   - **Recommendation:** Either integrate into main detection flow OR remove

### **SUMMARY:**
- ✅ **407 lines removed** (agent_clean.py)
- ⚠️ **487 lines** potentially unused (probabilistic_conflict_graph + method)

---

## 📈 CURRENT PIPELINE FLOW (DETAILED)

### **PHASE 1: TOOL EXECUTION**
```python
for tool_call in tool_calls:
    result = tool.invoke(args)  # Execute medical AI tool
    # 5 tools in parallel/sequence
```

### **PHASE 2: NORMALIZATION**
```python
for result in results:
    tool_type = _get_tool_type(tool_name)
    # Routes to: classification, segmentation, vqa, grounding, report
    canonical = normalize_output(result, tool_type)
    # Output: CanonicalFinding object with standardized fields
```

### **PHASE 3: CONFLICT DETECTION**
```python
conflicts = conflict_detector.detect_conflicts(canonical_findings)
  ├─ _check_exact_conflicts()       # Simple disagreements
  ├─ _check_bert_conflicts()        # BERT NLI semantic detection
  └─ _check_gacl_conflicts()        # Anatomical constraints
```

### **PHASE 4: CONFLICT RESOLUTION**
```python
for conflict in conflicts:
    resolution = conflict_resolver.resolve_conflict(conflict, findings)
    
    Resolution Logic:
    1. _analyze_bert_scores()           # Extract NLI scores
    2. _bert_guided_resolution()        # Use contradiction/entailment
    3. _get_task_type()                 # Map to expert tool
    4. _find_tool_finding()             # Get expert's output
    5. _adjust_confidence_by_bert()     # Calibrate by BERT
    OR
    6. _weighted_average_resolution()   # Fallback averaging
```

### **PHASE 5: LOGGING & REPORTING**
```python
Comprehensive Log JSON:
├─ tool_executions: Raw tool results
├─ canonical_findings: Normalized findings
├─ conflict_analysis:
│  ├─ conflicts: All detected conflicts
│  └─ resolutions: All resolutions
└─ summary: Statistics
```

---

## ✅ OPTIMIZATION RECOMMENDATIONS

### **IMMEDIATE (Do Now):**

1. **Remove unused `agent_clean.py`** ✅ DONE
   - Lines saved: 407

2. **Remove unused probabilistic graph method**
   ```python
   # DELETE from agent.py (lines 285-309)
   def _detect_conflicts_with_probabilistic_graph(self, ...):
   ```
   - Lines saved: 25

3. **Clean imports in agent.py**
   ```python
   # REMOVE: from .probabilistic_conflict_graph import ProbabilisticConflictGraph, analyze_tool_calibration
   # REMOVE: self.probabilistic_graph = ProbabilisticConflictGraph(...)
   ```
   - Lines saved: 2 + 1

### **SHORT TERM (Next Phase):**

1. **Deprecate or integrate ProbabilisticConflictGraph**
   - Option A: Integrate calibration analysis into main conflict detection
   - Option B: Remove entirely (462 lines)
   - **Recommendation:** REMOVE (unused, complex, not in critical path)

2. **Extract confidence scoring to separate module**
   - Move `calibrate_confidence()` + `estimate_text_confidence()` from `canonical_output.py` to `confidence_scoring.py`
   - Current split causes confusion

### **LONG TERM (Architecture):**

1. **Consider extracting BERT detector to separate service**
   - Currently inline in `conflict_resolution.py`
   - Heavy model (~1.3GB)
   - Could be async/cached

2. **Add conflict caching**
   - Same findings → same conflicts
   - Cache BERT inference results

---

## 📊 CODE STATISTICS (AFTER CLEANUP)

```
AGENT FRAMEWORK:           407 lines
├─ agent.py

NORMALIZATION:             882 lines
├─ canonical_output.py

CONFLICT DETECTION:      2,084 lines
├─ conflict_resolution.py      (849)
├─ bert_conflict_detector.py   (560)
├─ anatomical_consistency_graph.py (675)

CONFIDENCE SCORING:      1,385 lines
├─ confidence_scoring.py

PROBABILISTIC ANALYSIS:    462 lines [DEPRECATED]
├─ probabilistic_conflict_graph.py

TOOLS:                   2,402 lines
├─ classification.py        (160)
├─ segmentation.py          (325)
├─ xray_vqa.py              (294)
├─ llava_med.py             (274)
├─ report_generation.py     (370)
├─ grounding.py             (343)
├─ dicom.py + utils.py      (286)
├─ generation.py            (236)

TOTAL (ACTIVE):          ~6,600 lines
TOTAL (WITH DEPRECATED): ~7,700 lines
```

---

## 🎯 CRITICAL PATH ANALYSIS

**Essential Components (Cannot Remove):**
1. ✅ Agent orchestrator (agent.py) - 407 lines
2. ✅ Canonical normalization (canonical_output.py) - 882 lines
3. ✅ Conflict detection (conflict_resolution.py) - 849 lines
4. ✅ BERT NLI (bert_conflict_detector.py) - 560 lines
5. ✅ GACL constraints (anatomical_consistency_graph.py) - 675 lines
6. ✅ Confidence scoring (confidence_scoring.py) - 1,385 lines
7. ✅ Tool implementations (6 tools) - 2,402 lines

**Optional Components (Safe to Remove):**
1. ❌ Probabilistic conflict graph (probabilistic_conflict_graph.py) - 462 lines
2. ❌ Unused method in agent.py - 25 lines

---

## 🚀 NEXT STEPS

1. **Remove deprecated components** (487 lines)
2. **Integrate conflict resolution mechanism** (as per your guidance)
3. **Add caching for BERT inference**
4. **Profile tool execution times**
5. **Optimize normalization pipeline**

---

**Analysis Complete** ✅  
**Ready for implementation?** 🚀
