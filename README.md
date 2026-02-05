<h1 align="center">
🤖 MedRAX: Medical Reasoning Agent for Chest X-ray
</h1>
<p align="center"> <a href="https://arxiv.org/abs/2502.02673" target="_blank"><img src="https://img.shields.io/badge/arXiv-ICML 2025-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://github.com/bowang-lab/MedRAX"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a> <a href="https://huggingface.co/datasets/wanglab/chest-agent-bench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a> </p>

![](assets/demo_fast.gif?autoplay=1)

<br>

## 📋 Table of Contents

- [📋 Table of Contents](#-table-of-contents)
- [Abstract](#abstract)
- [MedRAX Overview](#medrax-overview)
  - [Integrated Tools](#integrated-tools)
- [ChestAgentBench](#chestagentbench)
- [Advanced: Conflict Detection \& Resolution Pipeline](#advanced-conflict-detection--resolution-pipeline)
  - [System Overview](#system-overview)
  - [Technical Foundation](#technical-foundation)
  - [1. Conflict Detection Engine](#1-conflict-detection-engine)
    - [**Multi-Method Detection Approach**](#multi-method-detection-approach)
      - [**a) BERT-Based Semantic NLI (Primary)**](#a-bert-based-semantic-nli-primary)
      - [**b) Rule-Based Confidence Gap Analysis (Fallback)**](#b-rule-based-confidence-gap-analysis-fallback)
      - [**c) Anatomical Consistency Validation**](#c-anatomical-consistency-validation)
  - [2. Confidence Calibration \& Fusion](#2-confidence-calibration--fusion)
  - [3. Argumentation Graphs](#3-argumentation-graphs)
  - [4. Tool Trust Management](#4-tool-trust-management)
  - [5. Intelligent Abstention Logic](#5-intelligent-abstention-logic)
  - [Full Conflict Resolution Workflow Example](#full-conflict-resolution-workflow-example)
  - [Configuration](#configuration)
  - [Conflict Resolution Outputs](#conflict-resolution-outputs)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Getting Started](#getting-started)
- [Tool Selection and Initialization](#tool-selection-and-initialization)
- [Automatically Downloaded Models](#automatically-downloaded-models)
  - [Classification Tool](#classification-tool)
  - [Segmentation Tool](#segmentation-tool)
  - [Grounding Tool](#grounding-tool)
  - [LLaVA-Med Tool](#llava-med-tool)
  - [Report Generation Tool](#report-generation-tool)
  - [Visual QA Tool](#visual-qa-tool)
  - [MedSAM Tool](#medsam-tool)
  - [Utility Tools](#utility-tools)
- [Manual Setup Required](#manual-setup-required)
  - [Image Generation Tool](#image-generation-tool)
- [Configuration Notes](#configuration-notes)
  - [Required Parameters](#required-parameters)
  - [Memory Management](#memory-management)
  - [Local LLMs](#local-llms)
  - [Optional: OpenAI-compatible Providers](#optional-openai-compatible-providers)
- [Star History](#star-history)
- [Authors \& Citation](#authors--citation)
  - [Authors](#authors)
- [Citation](#citation)

<br>

## Abstract
Chest X-rays (CXRs) play an integral role in driving critical decisions in disease management and patient care. While recent innovations have led to specialized models for various CXR interpretation tasks, these solutions often operate in isolation, limiting their practical utility in clinical practice. We present MedRAX, the first versatile AI agent that seamlessly integrates state-of-the-art CXR analysis tools and multimodal large language models into a unified framework. MedRAX dynamically leverages these models to address complex medical queries without requiring additional training. To rigorously evaluate its capabilities, we introduce ChestAgentBench, a comprehensive benchmark containing 2,500 complex medical queries across 7 diverse categories. Our experiments demonstrate that MedRAX achieves state-of-the-art performance compared to both open-source and proprietary models, representing a significant step toward the practical deployment of automated CXR interpretation systems.
<br><br>


## MedRAX Overview

MedRAX is built on a robust technical foundation:
- **Core Architecture**: Built on LangChain and LangGraph frameworks
- **Language Model**: Uses GPT-4o with vision capabilities as the backbone LLM
- **Deployment**: Supports both local and cloud-based deployments
- **Interface**: Production-ready interface built with Gradio
- **Modular Design**: Tool-agnostic architecture allowing easy integration of new capabilities

### Integrated Tools
- **Visual QA**: Utilizes CheXagent and LLaVA-Med for complex visual understanding and medical reasoning
- **Segmentation**: Employs MedSAM and PSPNet model trained on ChestX-Det for precise anatomical structure identification
- **Grounding**: Uses Maira-2 for localizing specific findings in medical images
- **Report Generation**: Implements SwinV2 Transformer trained on CheXpert Plus for detailed medical reporting
- **Disease Classification**: Leverages DenseNet-121 from TorchXRayVision for detecting 18 pathology classes
- **X-ray Generation**: Utilizes RoentGen for synthetic CXR generation
- **Utilities**: Includes DICOM processing, visualization tools, and custom plotting capabilities
<br><br>


## ChestAgentBench
We introduce ChestAgentBench, a comprehensive evaluation framework with 2,500 complex medical queries across 7 categories, built from 675 expert-curated clinical cases. The benchmark evaluates complex multi-step reasoning in CXR interpretation through:

- Detection
- Classification
- Localization
- Comparison
- Relationship
- Diagnosis
- Characterization

Download the benchmark: [ChestAgentBench on Hugging Face](https://huggingface.co/datasets/wanglab/chest-agent-bench)
```
huggingface-cli download wanglab/chestagentbench --repo-type dataset --local-dir chestagentbench
```

Unzip the Eurorad figures to your local `MedMAX` directory.
```
unzip chestagentbench/figures.zip
```

To evaluate with GPT-4o, set your OpenAI API key and run the quickstart script.
```
export OPENAI_API_KEY="<your-openai-api-key>"
python quickstart.py \
    --model chatgpt-4o-latest \
    --temperature 0.2 \
    --max-cases 2 \
    --log-prefix chatgpt-4o-latest \
    --use-urls
```

<br>

## Advanced: Conflict Detection & Resolution Pipeline

Medical image interpretation often results in conflicting outputs from specialized AI tools due to model bias, dataset differences, and task-specific training. MedRAX implements a sophisticated **multi-layer conflict detection and resolution system** that:

- **Detects contradictions** using BERT-based semantic NLI and confidence gap analysis
- **Validates anatomical consistency** to eliminate physically impossible interpretations  
- **Calibrates confidence scores** across heterogeneous tool outputs
- **Structures arguments** using formal argumentation graphs with explicit reasoning traces
- **Learns tool reliability** through persistent trust weight management
- **Intelligently abstains** when confidence is too low, flagging critical findings for human review

This ensures **clinical safety** and prevents potentially dangerous misdiagnoses in critical care scenarios.

### System Overview

The conflict management pipeline consists of **5 integrated, interconnected components**:

```
Tool Outputs (7+ models)
        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 1: CONFLICT DETECTION                         │
│  - Rule-based confidence gap analysis                │
│  - BERT-based semantic NLI (Natural Language Infr.)  │
│  - Anatomical consistency validation                 │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: CONFIDENCE CALIBRATION                     │
│  - Task-specific raw score extraction                │
│  - Min-max normalization across tools                │
│  - Isotonic regression / temperature scaling         │
│  - Cross-model fusion                                │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: ARGUMENTATION GRAPHS                       │
│  - Structure conflicts as explicit argument graphs   │
│  - Support vs attack edges with weighted strengths   │
│  - Cycle detection for circular logic                │
│  - Certainty scoring for decision clarity            │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 4: TOOL TRUST MANAGEMENT                      │
│  - Historical performance tracking (correct/total)   │
│  - Weighted voting based on tool reliability         │
│  - Persistent trust weights across sessions          │
│  - Real-time weight updates after resolution         │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 5: INTELLIGENT ABSTENTION                     │
│  - Detect when confidence is too low                 │
│  - Flag critical findings requiring human review     │
│  - Risk-aware decision thresholds                    │
│  - Clinical severity-based escalation                │
└─────────────────────────────────────────────────────┘
        ↓
Resolved Output + Confidence + Reasoning Trace
```

---

### Technical Foundation

**Problem Statement:** When multiple specialized models interpret the same CXR, they may disagree due to:
1. **Dataset bias**: Training on different patient populations or imaging protocols
2. **Task specialization**: Each model optimized for specific pathologies
3. **Architectural differences**: Different CNN/transformer backbones with different inductive biases
4. **Confidence calibration mismatch**: Raw scores not directly comparable across models

**Solution**: A formal multi-layer framework combining NLP, symbolic reasoning, and probabilistic inference.

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Conflict Detection** | Identify disagreements systematically | BERT-NLI + rule-based analysis |
| **Confidence Calibration** | Normalize heterogeneous scores | Isotonic regression + temperature scaling |
| **Argumentation Graphs** | Explain reasoning transparently | Weighted argumentation frameworks (AF) |
| **Trust Management** | Learn tool reliability over time | Bayesian performance tracking |
| **Abstention Logic** | Know when to defer to humans | Risk-aware thresholds + clinical severity |

---

### 1. Conflict Detection Engine

#### **Multi-Method Detection Approach**

##### **a) BERT-Based Semantic NLI (Primary)**

Uses state-of-the-art Natural Language Inference (NLI) to detect semantic conflicts in textual outputs from different tools:

```python
from medrax.agent.bert_conflict_detector import BERTConflictDetector

# Initialize detector with Microsoft DeBERTa-MNLI
detector = BERTConflictDetector(
    nli_model_name="microsoft/deberta-base-mnli",  # 433K NLI examples training
    conflict_threshold=0.7,
    device="cuda"
)

# Example: Detect contradiction between tool outputs
result = detector.detect_conflict(
    text1="Pneumothorax present, occupying 15% of hemithorax",
    text2="No pneumothorax detected in this study"
)

print(result.to_dict())
# Output:
# {
#     "has_conflict": True,
#     "conflict_probability": 0.92,           # 92% confidence = contradiction
#     "entailment_prob": 0.03,                # 3% = agreement
#     "neutral_prob": 0.05,                   # 5% = complementary info
#     "conflict_type": "contradiction",
#     "explanation": "Statement 2 directly contradicts Statement 1"
# }
```

**Technical Implementation:**
- **Model Architecture**: Microsoft DeBERTa (Decoding-enhanced BERT) with 433K MNLI + SNLI fine-tuning
- **Classification Scheme**:
  - `CONTRADICTION` (class 0): Tools strongly disagree → **Triggers conflict resolution**
  - `NEUTRAL` (class 1): Statements discuss different aspects → **No direct conflict**
  - `ENTAILMENT` (class 2): Tools essentially agree → **No conflict, can be fused**
- **Advantages**:
  - Catches semantic conflicts missed by simple string matching
  - Example: "Pneumothorax present" vs "Lung is collapsed" → Correctly identified as **agreement** (same finding, different terminology)
  - Example: "Mild edema" vs "No pulmonary edema" → Correctly identified as **contradiction**
  - Handles paraphrases, synonyms, and medical terminology variations

##### **b) Rule-Based Confidence Gap Analysis (Fallback)**

For faster detection when BERT overhead is unacceptable:

```python
# Automatic fallback when tools provide structured output
confidence_tool_a = 0.92  # "Cardiomegaly present"
confidence_tool_b = 0.15  # "Cardiomegaly absent"

confidence_gap = abs(confidence_tool_a - confidence_tool_b)
# Gap = 0.77 > THRESHOLD (0.4) → CONFLICT DETECTED

# Severity levels:
# - Critical: gap > 0.7 (e.g., 0.92 vs 0.05)
# - Moderate: gap 0.4-0.7 (e.g., 0.85 vs 0.25)
# - Minor: gap < 0.4 (e.g., 0.55 vs 0.25)
```

**Key Features:**
- **Speed**: O(1) complexity compared to BERT's O(n) encoding
- **Fallback Strategy**: Used when tools output numerical confidences without text
- **Severity Classification**: Enables risk-stratified conflict handling

##### **c) Anatomical Consistency Validation**

Validates if conflicting findings violate anatomical constraints (GACL - Graph-based Anatomical Consistency Logic):

```python
from medrax.agent.anatomical_consistency_graph import GACLConflictDetector

consistency_checker = GACLConflictDetector()

# Define findings with anatomical attributes
findings = {
    "pneumothorax": {
        "occupancy": "present",
        "aeration": "absent",
        "density": "air",
        "volume_change": "increased"
    },
    "pleural_effusion": {
        "occupancy": "present",
        "aeration": "decreased",
        "density": "fluid",
        "volume_change": "normal"
    }
}

# Check for anatomical incompatibilities
incompatibilities = consistency_checker.check_incompatibilities(findings)
# Returns list of findings that physically cannot coexist based on anatomy
```

**Universal Attribute Axes** (works for ANY pathology):
- **Occupancy**: Present / Absent
- **Aeration**: Normal / Decreased / Absent
- **Density**: Air / Fluid / Soft Tissue / Calcified
- **Volume**: Increased / Decreased / Normal
- **Mass Effect**: Shift / Compression / None

**Validation Logic:**
- Graph-based incompatibility checking (edges define valid relationships)
- Catches physically impossible combinations (e.g., pneumothorax + consolidated lung tissue in same space)
- Medical knowledge from radiology literature encoded as constraints

---

### 2. Confidence Calibration & Fusion

Different tools output confidence scores in different formats and scales. MedRAX normalizes and calibrates them using multiple techniques:

```python
from medrax.agent.confidence_scoring import ConfidenceScorer, TaskType

scorer = ConfidenceScorer()

# Tool outputs in heterogeneous formats
tool_outputs = [
    {
        "task_type": "classification",
        "model_name": "DenseNet-121",
        "raw_output": {
            "pneumothorax_prob": 0.89,
            "cardiomegaly_prob": 0.72
        }
    },
    {
        "task_type": "vqa",
        "model_name": "LLaVA-Med",
        "raw_output": "Pneumothorax is present with high confidence"
    },
    {
        "task_type": "grounding",
        "model_name": "Maira-2",
        "raw_output": 0.85  # Confidence in localization
    }
]

# Normalize to [0, 1] range
calibrated_scores = scorer.calibrate_outputs(tool_outputs)

print(calibrated_scores)
# {
#     "DenseNet-121": 0.89,      # Already [0,1], kept as-is
#     "LLaVA-Med": 0.87,         # Extracted from text, normalized
#     "Maira-2": 0.85            # Kept as-is
# }
```

**Calibration Methods**:

1. **Min-Max Normalization**: Scale raw outputs to [0, 1]
   - Formula: `normalized = (value - min) / (max - min)`
   - Applied when confidence bounds are known

2. **Isotonic Regression**: Learn perfect calibration from historical data
   - Non-parametric approach that monotonically maps raw scores to probabilities
   - Uses historical expert-verified resolutions to compute calibration curve
   - More accurate than temperature scaling for heterogeneous models

3. **Temperature Scaling**: Adjust confidence temperature for perfect calibration
   - Formula: `p_calibrated = softmax(logits / T)`
   - T > 1: soften predictions (reduce overconfidence)
   - T < 1: sharpen predictions (increase confidence spread)
   - Preserves ranking while improving calibration

4. **Confidence Fusion**: Weighted average using trust weights
   - Formula: `fused_confidence = Σ(weight_i × confidence_i) / Σ(weight_i)`
   - Weights learned from historical performance (see Layer 4)
   - Produces single calibrated confidence from multiple tool outputs

---

### 3. Argumentation Graphs

Structures conflicts as explicit argument graphs for **transparent reasoning and explainability**:

```python
from medrax.agent.argumentation_graph import ArgumentGraphBuilder

builder = ArgumentGraphBuilder()

# Build argument graph from conflicting tool outputs
graph = builder.build_from_conflict(
    claim="Pneumothorax is present",
    tools_involved=["DenseNet-121", "CheXagent", "LLaVA-Med"],
    confidences=[0.92, 0.15, 0.88],
    values=[True, False, True],
    tool_trust_weights={
        "DenseNet-121": 0.89,   # 89% historically accurate
        "CheXagent": 0.76,      # 76% historically accurate
        "LLaVA-Med": 0.82       # 82% historically accurate
    }
)

print(graph)
# ArgumentGraph(
#     claim='Pneumothorax is present',
#     support=[
#         ArgumentNode(DenseNet-121: 0.92 weighted=0.82),
#         ArgumentNode(LLaVA-Med: 0.88 weighted=0.72)
#     ],
#     attack=[
#         ArgumentNode(CheXagent: 0.15 weighted=0.11)
#     ],
#     support_strength=1.54,
#     attack_strength=0.11,
#     certainty=0.93,
#     net_winner="support"
# )
```

**Key Metrics**:
- **Support Strength**: Sum of confidences × trust weights from tools supporting the claim
- **Attack Strength**: Sum from tools attacking the claim
- **Certainty**: How dominant the winner is (0.0 = unclear, 1.0 = certain)
- **Net Winner**: "support", "attack", or "unclear"

**Explainability Output** (for radiologist review):
```
CLAIM: "Pneumothorax is present"

SUPPORT (83.1% probability):
  ✓ DenseNet-121: 92% confident (trust: 89%) → weighted strength: 0.82
  ✓ LLaVA-Med: 88% confident (trust: 82%) → weighted strength: 0.72
  Total support: 1.54

ATTACK (6.9% probability):
  ✗ CheXagent: 85% confident it's ABSENT (trust: 76%) → weighted strength: 0.11
  Total attack: 0.11

DECISION: Support overwhelmingly (certainty: 0.93)
RISK: Low - clear consensus among high-trust models
```

### 4. Tool Trust Management

Learns which tools are reliable over time:

```python
from medrax.agent.tool_trust import ToolTrustManager

# Initialize or load existing trust weights
trust_manager = ToolTrustManager(
    persistence_file="tool_trust_weights.json"
)

# Initialize tools with neutral trust
trust_manager.initialize_tool("DenseNet-121", initial_weight=1.0)
trust_manager.initialize_tool("LLaVA-Med", initial_weight=1.0)

# After expert reviews a resolved conflict:
# If DenseNet-121 was correct:
trust_manager.update_tool_trust("DenseNet-121", was_correct=True)
# DenseNet: 1 correct / 1 total = 1.0 weight

# If CheXagent was wrong:
trust_manager.update_tool_trust("CheXagent", was_correct=False)
# CheXagent: 0 correct / 1 total = 0.0 weight

# Get current trust profile
weights = trust_manager.get_all_weights()
# {
#     "DenseNet-121": 0.92,      # 92% accurate
#     "LLaVA-Med": 0.78,         # 78% accurate
#     "CheXagent": 0.71,         # 71% accurate
#     "Maira-2": 0.85            # 85% accurate
# }

# Weights automatically persist to JSON
# Survives application restarts
```

**Impact**: Higher-trust tools get more weight in voting and confidence fusion.

### 5. Intelligent Abstention Logic

Knows when to say "I don't know" instead of forcing a decision:

```python
from medrax.agent.abstention_logic import AbstentionLogic

abstention = AbstentionLogic(
    close_vote_thr=0.2,           # Abstain if gap < 20%
    uncertainty_thr=0.6,          # Abstain if certainty < 60%
    critical_certainty_thr=0.8,   # For life-threatening: need 80%
    min_tools=2                   # Need ≥2 tools reporting
)

# Check if we should abstain from making a decision
decision = abstention.should_abstain(
    support_strength=0.92,        # Tools supporting the claim
    attack_strength=0.78,         # Tools attacking the claim
    certainty=0.45,               # How certain are we?
    has_cycles=False,             # Any circular logic?
    clinical_severity="critical", # Life-threatening finding
    num_tools=3
)

print(decision.to_dict())
# {
#     "should_abstain": True,
#     "reason": "overall_confidence_too_low",
#     "confidence": 0.45,
#     "explanation": "Pneumothorax detection has low certainty (45%). For critical findings, we need 80%+. Recommend human radiologist review.",
#     "risk_level": "high"
# }
```

**Abstention Triggers**:
1. **Circular Logic**: Graph has cycles (tools creating circular justifications)
2. **Close Vote**: Support vs attack strengths are too similar
3. **High Uncertainty**: Overall certainty below threshold
4. **Critical Findings Unclear**: Life-threatening findings need higher confidence bar
5. **Insufficient Data**: Fewer than 2 tools reporting

### Full Conflict Resolution Workflow Example

```python
from medrax.agent.conflict_resolution import ConflictResolver

resolver = ConflictResolver(
    deferral_threshold=0.6,
    enable_bert_detection=True,
    enable_gacl_validation=True
)

# Scenario: Tools give conflicting CXR interpretations
tool_results = {
    "DenseNet-121": {
        "cardiomegaly": {"confidence": 0.89, "present": True}
    },
    "CheXagent": {
        "cardiomegaly": {"confidence": 0.25, "present": False}
    },
    "LLaVA-Med": {
        "report": "Heart size is enlarged, consistent with cardiomegaly"
    }
}

# Run full conflict resolution pipeline
resolution = resolver.resolve_conflicts(tool_results)

print(resolution)
# {
#     "finding": "Cardiomegaly",
#     "resolved_value": True,
#     "confidence": 0.88,
#     "method": "argumentation_graph_with_bert_validation",
#     "bert_conflict_score": 0.91,  # BERT detected contradiction
#     "support_tools": ["DenseNet-121", "LLaVA-Med"],
#     "attack_tools": ["CheXagent"],
#     "reasoning": "DenseNet-121 and LLaVA-Med strongly support cardiomegaly...",
#     "should_defer_to_human": False,
#     "risk_level": "low"
# }

# If should_defer_to_human == True:
# → Result marked for expert radiologist review
# → Confidence capped at deferral_threshold
# → Human feedback used to update trust weights
```

### Configuration

Enable/disable conflict resolution in the agent:

```python
from medrax.agent.agent import Agent

agent = Agent(
    model=llm,
    tools=tools,
    enable_conflict_resolution=True,      # Enable pipeline
    conflict_sensitivity=0.4,              # Detection sensitivity [0-1]
    deferral_threshold=0.6                 # Certainty threshold for human review
)

# Log conflicts to file for analysis
# Logs saved to: logs/conflict_resolution_TIMESTAMP.json
```

### Conflict Resolution Outputs

All conflicts are logged with full traceability:

```json
{
  "timestamp": "2025-02-06T10:30:45.123Z",
  "case_id": "CXR_001",
  "findings": [
    {
      "finding": "Pneumothorax",
      "initial_values": ["present", "absent", "present"],
      "initial_confidences": [0.92, 0.15, 0.88],
      "conflict_type": "presence",
      "conflict_severity": "critical",
      "bert_analysis": {
        "contradiction_prob": 0.87,
        "entailment_prob": 0.08,
        "neutral_prob": 0.05
      },
      "resolved_value": "present",
      "resolved_confidence": 0.90,
      "reasoning_chain": [
        "BERT detected high contradiction probability (0.87)",
        "Argumentation graph: support_strength=1.80 > attack_strength=0.15",
        "Anatomical validation: pneumothorax and effusion compatible",
        "Trust-weighted voting: DenseNet (0.89) + LLaVA-Med (0.82) >> CheXagent (0.76)"
      ],
      "human_review_required": false,
      "risk_assessment": "low"
    }
  ]
}
```

<br>

## Installation
### Prerequisites
- Python 3.8+
- CUDA/GPU for best performance

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/bowang-lab/MedRAX.git
cd MedRAX

# Install package
pip install -e .
```

### Getting Started
```bash
# Start the Gradio interface
python main.py
```
or if you run into permission issues
```bash
sudo -E env "PATH=$PATH" python main.py
```
You need to setup the `model_dir` inside `main.py` to the directory where you want to download or already have the weights of above tools from Hugging Face.
Comment out the tools that you do not have access to.
Make sure to setup your OpenAI API key in `.env` file!
<br><br><br>


## Tool Selection and Initialization

MedRAX supports selective tool initialization, allowing you to use only the tools you need. Tools can be specified when initializing the agent (look at `main.py`):

```python
selected_tools = [
    "ImageVisualizerTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    # Add or remove tools as needed
]

agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights"
)
```

<br><br>
## Automatically Downloaded Models

The following tools will automatically download their model weights when initialized:

### Classification Tool
```python
ChestXRayClassifierTool(device=device)
```

### Segmentation Tool
```python
ChestXRaySegmentationTool(device=device)
```

### Grounding Tool
```python
XRayPhraseGroundingTool(
    cache_dir=model_dir, 
    temp_dir=temp_dir, 
    load_in_8bit=True, 
    device=device
)
```
- Maira-2 weights download to specified `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### LLaVA-Med Tool
```python
LlavaMedTool(
    cache_dir=model_dir, 
    device=device, 
    load_in_8bit=True
)
```
- Automatic weight download to `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### Report Generation Tool
```python
ChestXRayReportGeneratorTool(
    cache_dir=model_dir, 
    device=device
)
```

### Visual QA Tool
```python
XRayVQATool(
    cache_dir=model_dir, 
    device=device
)
```
- CheXagent weights download automatically

### MedSAM Tool
```
Support for MedSAM segmentation will be added in a future update.
```

### Utility Tools
No additional model weights required:
```python
ImageVisualizerTool()
DicomProcessorTool(temp_dir=temp_dir)
```
<br>

## Manual Setup Required

### Image Generation Tool
```python
ChestXRayGeneratorTool(
    model_path=f"{model_dir}/roentgen", 
    temp_dir=temp_dir, 
    device=device
)
```
- RoentGen weights require manual setup:
  1. Contact authors: https://github.com/StanfordMIMI/RoentGen
  2. Place weights in `{model_dir}/roentgen`
  3. Optional tool, can be excluded if not needed
<br>

## Configuration Notes

### Required Parameters
- `model_dir` or `cache_dir`: Base directory for model weights that Hugging Face uses
- `temp_dir`: Directory for temporary files
- `device`: "cuda" for GPU, "cpu" for CPU-only

### Memory Management
- Consider selective tool initialization for resource constraints
- Use 8-bit quantization where available
- Some tools (LLaVA-Med, Grounding) are more resource-intensive
<br>

### Local LLMs
If you are running a local LLM using frameworks like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/), you need to configure your environment variables accordingly. For example:
```
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```
<br>

### Optional: OpenAI-compatible Providers

MedRAX supports OpenAI-compatible APIs, allowing regional or local LLM providers to serve as alternative backends.

For example, to use **Qwen3-VL** via [Alibaba Cloud DashScope](https://bailian.console.aliyun.com/?tab=model#/model-market), set the following environment variables:

```bash
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="<your-dashscope-api-key>"
export OPENAI_MODEL="qwen3-vl-235b-a22b-instruct"
```
<br>

## Star History
<div align="center">
  
[![Star History Chart](https://api.star-history.com/svg?repos=bowang-lab/MedRAX&type=Date)](https://star-history.com/#bowang-lab/MedRAX&Date)

</div>
<br>


## Authors & Citation

### Authors
- **Adibvafa Fallahpour**¹²³⁴ * (adibvafa.fallahpour@mail.utoronto.ca)
- ****Jun Ma****²³ *
- **Alif Munim**³⁵ *
- ****Hongwei Lyu****³
- ****Bo Wang****¹²³⁶

¹ Department of Computer Science, University of Toronto, Toronto, Canada <br>
² Vector Institute, Toronto, Canada <br>
³ University Health Network, Toronto, Canada <br>
⁴ Cohere, Toronto, Canada <br>
⁵ Cohere Labs, Toronto, Canada <br>
⁶ Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada

<br>
* Equal contribution
<br><br>


## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{fallahpour2025medraxmedicalreasoningagent,
      title={MedRAX: Medical Reasoning Agent for Chest X-ray}, 
      author={Adibvafa Fallahpour and Jun Ma and Alif Munim and Hongwei Lyu and Bo Wang},
      year={2025},
      eprint={2502.02673},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.02673}, 
}
```

---
<p align="center">
Made with ❤️ at University of Toronto, Vector Institute, and University Health Network
</p>
