<h1 align="center">
🤖 MedRAX: Medical Reasoning Agent for Chest X-ray
</h1>
<p align="center"> <a href="https://arxiv.org/abs/2502.02673" target="_blank"><img src="https://img.shields.io/badge/arXiv-ICML 2025-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://github.com/bowang-lab/MedRAX"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a> <a href="https://huggingface.co/datasets/wanglab/chest-agent-bench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a> </p>

![](assets/demo_fast.gif?autoplay=1)

<br>

## 📋 Table of Contents

- [Abstract](#abstract)
- [MedRAX Overview](#medrax-overview)
- [ChestAgentBench](#chestagentbench)
- [Conflict Detection & Resolution Pipeline](#advanced-conflict-detection--resolution-pipeline)
  - [Motivation](#motivation)
  - [System Architecture](#system-architecture)
  - [Layer 1 — Conflict Detection](#layer-1-conflict-detection)
  - [Layer 2 — Confidence Calibration and Fusion](#layer-2-confidence-calibration-and-fusion)
  - [Layer 3 — Argumentation Graphs](#layer-3-argumentation-graphs)
  - [Layer 4 — Tool Trust Management](#layer-4-tool-trust-management)
  - [Layer 5 — Intelligent Abstention](#layer-5-intelligent-abstention)
  - [End-to-End Pipeline Integration](#end-to-end-pipeline-integration)
  - [Design Principles](#design-principles)
- [Installation](#installation)
- [Tool Selection and Initialization](#tool-selection-and-initialization)
- [Model Management](#automatically-downloaded-models)
- [Configuration Notes](#configuration-notes)
- [Authors & Citation](#authors--citation)

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

### Motivation

When multiple specialized AI models interpret the same chest X-ray, disagreements are inevitable. Each model carries inherent biases from its training data, architecture, and optimization objective. A DenseNet-121 classifier trained on NIH ChestX-ray14 may flag cardiomegaly with 89% confidence, while a CheXagent visual question-answering model trained on MIMIC-CXR reports the heart size as normal. Neither model is universally wrong—they simply encode different priors from different patient populations, labelling conventions, and learning paradigms.

In clinical practice, such contradictions are dangerous. A false negative on a tension pneumothorax or a missed cardiomegaly in a heart failure patient can delay life-saving intervention. Conversely, a false positive may trigger unnecessary procedures, increasing patient morbidity and healthcare costs. MedRAX addresses this fundamental challenge through a principled, five-layer conflict detection and resolution pipeline that systematically identifies disagreements, quantifies uncertainty, structures evidence, and—when the evidence is insufficient—defers to human expertise rather than forcing an unreliable decision.

---

### System Architecture

The pipeline processes tool outputs through five sequential, tightly integrated layers. Each layer refines the decision state before passing it to the next, producing a final resolved output accompanied by a calibrated confidence score and a complete reasoning trace.

```
Tool Outputs (7+ specialized models)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Layer 1 ─ CONFLICT DETECTION                        │
│  Semantic NLI · Confidence Gap · Anatomical GACL     │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Layer 2 ─ CONFIDENCE CALIBRATION & FUSION           │
│  Task-specific extraction · Normalization ·           │
│  Isotonic regression · Temperature scaling            │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Layer 3 ─ ARGUMENTATION GRAPHS                      │
│  Support / Attack edges · Trust-weighted strengths ·  │
│  Cycle detection · Certainty scoring                  │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Layer 4 ─ TOOL TRUST MANAGEMENT                     │
│  Historical accuracy tracking · Bayesian updating ·   │
│  Persistent weights · Weighted voting                 │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Layer 5 ─ INTELLIGENT ABSTENTION                    │
│  Risk-aware thresholds · Clinical severity gates ·    │
│  Cycle-based rejection · Human deferral               │
└──────────────────────────────────────────────────────┘
        │
        ▼
  Resolved Finding + Calibrated Confidence + Reasoning Trace
```

---

### Layer 1: Conflict Detection

The first layer employs a multi-method detection strategy that combines deep semantic understanding with efficient numerical heuristics and domain-specific anatomical knowledge. Findings from all active tools are grouped by pathology, and every pair of tool outputs within the same pathology group is examined for disagreement.

#### 1.1 BERT-Based Semantic Natural Language Inference

The primary detection mechanism leverages a fine-tuned DeBERTa model (He et al., 2021) trained on 433K examples from the Multi-Genre Natural Language Inference (MultiNLI) and Stanford Natural Language Inference (SNLI) corpora. Given two textual statements produced by different tools, the model classifies their relationship into one of three categories:

| Label | Interpretation | Action |
|-------|---------------|--------|
| **Contradiction** | The statements are mutually exclusive | Conflict flagged; resolution pipeline activated |
| **Neutral** | The statements address different aspects of the image | No conflict; findings are complementary |
| **Entailment** | The statements convey the same clinical meaning | No conflict; findings can be directly fused |

This approach captures semantic equivalences that surface-level string matching would miss. For example, "pneumothorax present" and "the right lung is collapsed" describe the same clinical finding using entirely different vocabulary; the NLI model correctly identifies this as entailment rather than a spurious conflict. Conversely, "mild pulmonary edema" and "no evidence of pulmonary edema" are correctly classified as a contradiction despite sharing most of their surface tokens.

The model outputs a probability distribution over all three classes. A conflict is declared when the contradiction probability exceeds a configurable threshold (default: 0.7), and the severity is graded as *critical* (>0.85), *moderate* (0.70–0.85), or *minor* (<0.70) to enable risk-stratified downstream handling.

#### 1.2 Rule-Based Confidence Gap Analysis

When tools produce structured numerical outputs rather than free text—for instance, a classification model returning a probability vector—a lightweight rule-based detector serves as a complementary fallback. The method computes the absolute difference between confidence scores assigned by different tools to the same pathology. If the gap exceeds a sensitivity threshold (default: 0.4), a conflict is flagged.

Severity is assigned proportionally: a gap exceeding 0.7 (e.g., one tool reporting 92% probability of cardiomegaly while another reports 15%) is classified as critical, gaps between 0.4 and 0.7 as moderate, and gaps below 0.4 as minor. This method operates in O(1) time per tool pair, making it suitable as a real-time screening layer when BERT inference latency is a concern.

#### 1.3 Anatomical Consistency Validation (GACL)

The third detection method applies domain-specific medical knowledge through Graph-based Anatomical Consistency Logic (GACL). Rather than relying on disease-specific rules, GACL encodes universal radiological attribute axes that generalize across all CXR findings:

| Attribute | Values | Clinical Meaning |
|-----------|--------|-----------------|
| **Occupancy** | Present, Absent | Whether a finding exists in a given region |
| **Aeration** | Normal, Decreased, Absent | Degree of air content in lung parenchyma |
| **Density** | Air, Fluid, Soft Tissue, Calcified | Radiodensity of the abnormality |
| **Volume** | Increased, Decreased, Normal | Size relative to anatomical norm |
| **Mass Effect** | Shift, Compression, None | Impact on adjacent structures |

These axes are connected through an incompatibility graph whose edges encode physically impossible combinations derived from radiology literature. For example, a region simultaneously exhibiting increased aeration (air trapping) and fluid-density opacity violates basic physics and is flagged as anatomically inconsistent. This layer catches conflicts that are invisible to both statistical NLI and simple numerical comparison—cases where two findings may each be individually plausible but cannot coexist in the same anatomical space.

---

### Layer 2: Confidence Calibration and Fusion

The heterogeneous tools integrated into MedRAX produce confidence scores in fundamentally different formats and scales. A DenseNet-121 classifier outputs sigmoid probabilities in [0, 1], a LLaVA-Med visual question-answering model embeds confidence linguistically ("high confidence," "likely present"), and a Maira-2 grounding model returns bounding-box localization scores on an arbitrary scale. Direct comparison of these raw scores is statistically invalid.

MedRAX addresses this through a model-agnostic confidence scoring pipeline implemented via task-specific extractors. Each tool type has a dedicated extractor that maps its native output format onto a standardized `ConfidenceResult` schema containing a raw score, a calibrated score, and an uncertainty estimate.

Calibration proceeds through three complementary methods:

**Min-Max Normalization** maps raw scores to the [0, 1] interval when the output bounds are known, preserving ordinal ranking while enabling cross-model comparison.

**Isotonic Regression** learns a non-parametric, monotonically non-decreasing mapping from raw scores to true probabilities using historically verified resolutions as ground truth. Unlike parametric methods, isotonic regression makes no distributional assumptions and adapts to each tool's unique miscalibration pattern—particularly important when different models exhibit systematically different calibration errors.

**Temperature Scaling** applies a single learned parameter *T* to the model's logits before softmax normalization. When *T* > 1, the distribution is softened, correcting for overconfident models; when *T* < 1, the distribution is sharpened, correcting for underconfident models. This method preserves the model's internal ranking while adjusting the magnitude of expressed confidence.

After calibration, scores from multiple tools are fused into a single per-finding confidence estimate via a trust-weighted average: *C*_fused = Σ(*w*_*i* × *c*_*i*) / Σ(*w*_*i*), where *w*_*i* is the trust weight for tool *i* (see Layer 4) and *c*_*i* is its calibrated confidence. This formulation naturally privileges historically reliable tools while still incorporating evidence from all sources.

---

### Layer 3: Argumentation Graphs

Once conflicts are detected and confidence scores are calibrated, the pipeline structures the disagreement as a formal weighted argumentation framework. For each contested finding (e.g., "Cardiomegaly is present"), an argument graph is constructed in which:

- **Support nodes** represent tools whose calibrated confidence exceeds 0.5 for the finding, indicating agreement with the claim. Each node carries a *strength* value computed as the product of the tool's calibrated confidence and its trust weight.
- **Attack nodes** represent tools whose calibrated confidence falls below 0.5, indicating disagreement. Their strength is computed analogously.
- **Edges** connect nodes to the central claim, forming a bipartite support/attack structure.

The graph produces four key metrics that drive downstream resolution:

**Support Strength** is the sum of weighted strengths across all supporting tools. **Attack Strength** is the corresponding sum for opposing tools. The **Confidence Gap** is the absolute difference between support and attack strengths, indicating how decisive the evidence is. **Certainty** normalizes the gap by total strength, producing a score in [0, 1] where values near 1.0 indicate overwhelming consensus and values near 0.0 indicate an evenly split decision.

A critical safety feature is **cycle detection**. In adversarial or degenerate cases, circular dependencies may arise where tool A supports tool B which supports tool C which contradicts tool A. The graph builder performs a topological analysis to identify such cycles, which are flagged as grounds for automatic abstention (see Layer 5).

The argumentation graph also serves as the primary **explainability mechanism**. For each resolved conflict, the graph is serialized into a human-readable reasoning trace that a reviewing radiologist can inspect: which tools supported the final decision, with what strength, and why opposing evidence was discounted. This transparency is essential for clinical trust and regulatory compliance.

---

### Layer 4: Tool Trust Management

Not all tools are equally reliable across all pathologies and clinical scenarios. MedRAX learns tool-specific trust weights through a persistent, incrementally updated tracking system. Each tool maintains a record of its historical performance: the number of cases in which its output was verified as correct versus incorrect by expert review or downstream validation.

The trust weight for tool *i* is computed as: *w*_*i* = *n*_correct / *n*_total, where *n*_correct is the cumulative count of verified correct predictions and *n*_total is the total number of evaluated predictions. Tools are initialized with a neutral prior (default weight of 1.0, backed by 10 pseudo-observations) to avoid cold-start instability while allowing rapid adaptation as real performance data accumulates.

Trust weights influence the pipeline at two critical points. First, they modulate the strength of each tool's contribution in the argumentation graph (Layer 3), ensuring that historically unreliable tools exert less influence on conflict resolution. Second, they weight the confidence fusion formula (Layer 2), so that the fused confidence score naturally gravitates toward the estimates of trusted tools.

Weights are persisted to disk as a JSON file and survive application restarts, enabling continuous learning across sessions. After each conflict resolution, the trust manager receives feedback on which tools were ultimately correct, updates their records, and saves the updated weights. Over time, this creates an adaptive system that self-corrects: if a tool's performance degrades due to distribution shift or model drift, its trust weight automatically decreases, reducing its impact on future decisions.

---

### Layer 5: Intelligent Abstention

The final layer addresses a fundamental requirement of clinical AI systems: the ability to recognize the limits of its own competence. Rather than forcing a resolution when the evidence is ambiguous or insufficient, MedRAX implements a principled abstention mechanism that defers to human expertise under specific, well-defined conditions.

Abstention is triggered by any of the following five conditions:

**Insufficient Evidence.** Fewer than two tools have reported on the finding in question. A single-tool opinion provides no basis for conflict assessment or confidence fusion and is flagged for human review.

**Circular Logic.** The argumentation graph contains cycles, indicating that the tools' positions form a logically incoherent structure. No reliable resolution can be derived from circular evidence.

**Close Vote.** The gap between support strength and attack strength falls below a configurable threshold (default: 0.2). When evidence is nearly evenly split, any resolution would be statistically indistinguishable from a coin flip.

**High Uncertainty.** The overall certainty score from the argumentation graph falls below a minimum confidence threshold (default: 0.6). This catches cases where many tools report but none with strong conviction.

**Critical Finding with Insufficient Certainty.** For life-threatening findings—tension pneumothorax, massive pleural effusion, acute pulmonary edema—the certainty threshold is elevated (default: 0.8). The clinical cost of a wrong decision on such findings is asymmetrically high, and the system applies a correspondingly higher evidentiary bar before committing to a resolution.

When abstention is triggered, the system returns a structured deferral response containing the reason for abstention, the current confidence level, a risk assessment (low/medium/high), and a human-readable explanation. The finding is explicitly flagged for radiologist review. Critically, human feedback on deferred cases is fed back into the trust management system (Layer 4), creating a closed learning loop that improves future resolution quality.

---

### End-to-End Pipeline Integration

The five layers operate as a single, integrated pipeline rather than as isolated modules. When the agent receives outputs from its constituent tools, the flow proceeds as follows:

1. All tool outputs are first converted into a unified canonical schema (`CanonicalFinding`) that standardizes pathology names, anatomical regions, confidence scores, and evidence types across all tool formats.

2. The conflict detection engine (Layer 1) groups findings by pathology and applies BERT-NLI, confidence gap analysis, and GACL validation in sequence. Each detected conflict is annotated with its type (presence, location, severity, or semantic), the tools involved, and a severity grade.

3. For each conflict, the confidence calibration pipeline (Layer 2) normalizes all involved tools' scores to a common scale using the appropriate task-specific extractor and calibration method.

4. The argumentation graph builder (Layer 3) constructs a support/attack graph using the calibrated confidences and current trust weights, computing strength totals, certainty, and checking for cycles.

5. The abstention logic (Layer 5) evaluates whether the graph's certainty, vote margin, and clinical context meet the minimum thresholds for automated resolution. If not, the finding is deferred with a full explanation.

6. If resolution proceeds, a fallback hierarchy determines the winner: BERT-guided resolution for high-confidence contradictions, task-specific expertise arbitration (e.g., trusting the classifier over the VQA model for binary presence detection), or trust-weighted voting as a last resort.

7. The trust manager (Layer 4) receives the resolution outcome. When expert feedback is available, it updates the involved tools' trust weights and persists them for future sessions.

8. The complete resolution—including the finding, resolved value, calibrated confidence, argumentation graph, reasoning chain, and any abstention flags—is logged to a timestamped JSON file for audit, analysis, and regulatory traceability.

---

### Design Principles

The conflict resolution pipeline is guided by several principles drawn from clinical decision support literature:

**Safety over accuracy.** The system is designed to err on the side of caution. When in doubt, it abstains and defers to a human rather than committing to a potentially harmful decision. The asymmetric certainty thresholds for critical findings reflect this philosophy.

**Transparency over opacity.** Every resolution is accompanied by a complete reasoning trace, from raw tool outputs through calibration, argumentation, and final decision. This enables radiologists to verify the system's logic and builds the trust necessary for clinical adoption.

**Adaptivity over rigidity.** The trust management and calibration systems learn continuously from resolved cases, adapting to changes in tool performance, patient population, and imaging protocols without requiring retraining or manual reconfiguration.

**Generality over specificity.** The GACL anatomical consistency framework uses universal attribute axes rather than disease-specific rules, enabling it to validate findings for any CXR pathology—including novel conditions not present in the training data—without modification.

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
