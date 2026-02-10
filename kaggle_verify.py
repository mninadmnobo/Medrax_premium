"""
MedRax Premium - Complete Kaggle Verification
Checks EVERYTHING: images, pipeline, modules, tools, dataset integrity, API, GPU.
Run this ONCE after setup to verify the environment is ready.
"""

import sys
import os
import json
import importlib
from pathlib import Path
from collections import Counter

TOTAL_CHECKS = 14
issues = []
warnings = []
stats = {}

def ok(msg):
    print(f"  ✅ {msg}")

def fail(msg):
    print(f"  ❌ {msg}")
    issues.append(msg)

def warn(msg):
    print(f"  ⚠️  {msg}")
    warnings.append(msg)

def info(msg):
    print(f"  ℹ️  {msg}")

print("=" * 70)
print("  MedRax Premium - Complete Kaggle Verification")
print("=" * 70)

# ============================================================
# 1. PYTHON VERSION
# ============================================================
print(f"\n[1/{TOTAL_CHECKS}] Python Version")
print(f"  Version: {sys.version.split()[0]}")
if sys.version_info >= (3, 10):
    ok("Python >= 3.10")
else:
    fail("Need Python >= 3.10")

# ============================================================
# 2. GPU
# ============================================================
print(f"\n[2/{TOTAL_CHECKS}] GPU")
gpu_ok = False
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        ok(f"{gpu_name} ({gpu_mem:.1f} GB)")
        stats["gpu"] = f"{gpu_name} ({gpu_mem:.1f} GB)"
        gpu_ok = True
    else:
        fail("No GPU! Enable: Settings > Accelerator > GPU T4 x2")
except ImportError:
    fail("PyTorch not installed")

# ============================================================
# 3. PACKAGES
# ============================================================
print(f"\n[3/{TOTAL_CHECKS}] Packages")
critical_packages = {
    "torch": "PyTorch", "torchvision": "TorchVision", "transformers": "Transformers",
    "langchain_core": "LangChain Core", "langchain_openai": "LangChain OpenAI",
    "langgraph": "LangGraph", "openai": "OpenAI SDK", "sklearn": "Scikit-learn",
    "PIL": "Pillow", "cv2": "OpenCV", "numpy": "NumPy", "pandas": "Pandas",
    "torchxrayvision": "TorchXRayVision", "pydicom": "PyDICOM", "einops": "Einops",
    "accelerate": "Accelerate", "peft": "PEFT", "bitsandbytes": "BitsAndBytes",
    "timm": "TIMM", "diffusers": "Diffusers", "dotenv": "python-dotenv",
    "requests": "Requests", "datasets": "HF Datasets", "tenacity": "Tenacity",
}
pkg_ok = 0
pkg_fail = 0
missing_pkgs = []
for pkg, desc in critical_packages.items():
    try:
        importlib.import_module(pkg)
        pkg_ok += 1
    except ImportError:
        pkg_fail += 1
        missing_pkgs.append(pkg)

if pkg_fail == 0:
    ok(f"{pkg_ok}/{pkg_ok} packages installed")
else:
    fail(f"{pkg_fail} packages missing: {', '.join(missing_pkgs)}")
    info("Run: pip install -r requirements.txt")
stats["packages"] = f"{pkg_ok}/{pkg_ok + pkg_fail}"

# ============================================================
# 4. FIND MEDRAX ROOT
# ============================================================
print(f"\n[4/{TOTAL_CHECKS}] MedRax Project Location")
search_paths = [
    "/kaggle/working",
    "/kaggle/working/Medrax_premium",
    "/kaggle/input/medrax-premium",
    os.getcwd(),
]
medrax_root = None
for path in search_paths:
    if os.path.exists(os.path.join(path, "medrax", "agent", "agent.py")):
        medrax_root = path
        break
if not medrax_root:
    for root, dirs, files in os.walk("/kaggle"):
        if "medrax" in dirs and os.path.exists(os.path.join(root, "medrax", "agent", "agent.py")):
            medrax_root = root
            break

if medrax_root:
    if medrax_root not in sys.path:
        sys.path.insert(0, medrax_root)
    ok(f"Found at: {medrax_root}")
    stats["medrax_root"] = medrax_root
else:
    fail("MedRax project NOT found! Upload code to Kaggle.")
    stats["medrax_root"] = None

# ============================================================
# 5. TRANSFORMERS COMPATIBILITY CHECK
# ============================================================
print(f"\n[5/{TOTAL_CHECKS}] Transformers Compatibility")
llava_content = ""

# Check if the fix is in place (should be in medrax/llava/model/__init__.py)
try:
    import transformers.utils
    if hasattr(transformers.utils, 'add_model_info_to_auto_map'):
        ok("transformers.utils.add_model_info_to_auto_map available")
    else:
        fail("add_model_info_to_auto_map missing - __init__.py fix not applied")
        info("Fix medrax/llava/model/__init__.py or apply monkey-patch in notebook setup cell")
except Exception as e:
    fail(f"Could not check transformers: {e}")

if medrax_root:
    # Check __init__.py has the permanent fix
    init_path = os.path.join(medrax_root, "medrax", "llava", "model", "__init__.py")
    if os.path.exists(init_path):
        with open(init_path) as f:
            init_content = f.read()
        if "add_model_info_to_auto_map" in init_content:
            ok("__init__.py has permanent transformers fix")
        else:
            warn("__init__.py missing permanent fix - apply via notebook setup cell")
    else:
        warn(f"__init__.py not found at {init_path}")

    # Check llava_med.py lazy imports
    llava_path = os.path.join(medrax_root, "medrax", "tools", "llava_med.py")
    if os.path.exists(llava_path):
        with open(llava_path) as f:
            llava_content = f.read()
        if "_lazy_load_llava" in llava_content:
            ok("llava_med.py has lazy imports")
        else:
            warn("llava_med.py uses eager imports")
    else:
        fail(f"llava_med.py not found at {llava_path}")
else:
    fail("Cannot check - MedRax root not found")

# ============================================================
# 6. CORE MODULE IMPORTS
# ============================================================
print(f"\n[6/{TOTAL_CHECKS}] Core Module Imports")
import_checks = {
    "medrax.agent.Agent": None,
    "medrax.agent.ConflictDetector": None,
    "medrax.agent.ConflictResolver": None,
    "medrax.agent.ArgumentGraphBuilder": None,
    "medrax.agent.ToolTrustManager": None,
    "medrax.agent.AbstentionLogic": None,
    "medrax.agent.CanonicalFinding": None,
    "medrax.agent.ConfidenceScoringPipeline": None,
    "medrax.utils.load_prompts_from_file": None,
}
agent_ok = 0
agent_fail = 0
for full_name in import_checks:
    module_path, class_name = full_name.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
        getattr(mod, class_name)
        import_checks[full_name] = True
        agent_ok += 1
    except Exception as e:
        import_checks[full_name] = str(e)
        agent_fail += 1

if agent_fail == 0:
    ok(f"All {agent_ok} agent/premium imports OK")
else:
    for name, result in import_checks.items():
        if result is not True:
            fail(f"{name}: {result}")
stats["agent_imports"] = f"{agent_ok}/{agent_ok + agent_fail}"

# ============================================================
# 7. TOOLS MODULE IMPORT
# ============================================================
print(f"\n[7/{TOTAL_CHECKS}] Tools Module Import")
tools_list = [
    "ChestXRayClassifierTool", "ChestXRaySegmentationTool", "XRayVQATool",
    "LlavaMedTool", "XRayPhraseGroundingTool", "ChestXRayReportGeneratorTool",
    "ChestXRayGeneratorTool", "DicomProcessorTool", "ImageVisualizerTool",
]
tools_ok = 0
tools_fail = 0
tools_errors = []
for tool_name in tools_list:
    try:
        mod = importlib.import_module("medrax.tools")
        getattr(mod, tool_name)
        tools_ok += 1
    except Exception as e:
        tools_fail += 1
        tools_errors.append(f"{tool_name}: {e}")

if tools_fail == 0:
    ok(f"All {tools_ok} tool classes importable")
else:
    fail(f"{tools_fail}/{tools_ok + tools_fail} tools failed to import")
    for err in tools_errors[:3]:
        info(f"  {err[:120]}")
    if "_lazy_load_llava" not in llava_content:
        info("This may be the transformers compatibility issue - check setup cell")
stats["tools_imports"] = f"{tools_ok}/{tools_ok + tools_fail}"

# ============================================================
# 8. DATASET: metadata.jsonl
# ============================================================
print(f"\n[8/{TOTAL_CHECKS}] Dataset: metadata.jsonl")
metadata_search = [
    "chestagentbench/metadata.jsonl",
    "/kaggle/working/chestagentbench/metadata.jsonl",
]
if medrax_root:
    metadata_search.append(os.path.join(medrax_root, "chestagentbench", "metadata.jsonl"))

metadata_path = None
for mp in metadata_search:
    if os.path.exists(mp):
        metadata_path = mp
        break

if metadata_path:
    with open(metadata_path) as f:
        lines = f.readlines()
    question_count = len(lines)
    size_mb = os.path.getsize(metadata_path) / (1024 * 1024)
    ok(f"{question_count} questions ({size_mb:.1f} MB)")
    stats["questions"] = question_count

    # Validate structure
    first = json.loads(lines[0])
    required_fields = ["question", "answer", "images", "case_id", "question_id", "categories"]
    missing_fields = [f for f in required_fields if f not in first]
    if missing_fields:
        fail(f"Missing fields in metadata: {missing_fields}")
    else:
        ok(f"All required fields present")

    # Count categories
    all_cats = []
    for line in lines:
        entry = json.loads(line)
        all_cats.extend(entry.get("categories", "").split(","))
    cat_counts = Counter(c.strip() for c in all_cats if c.strip())
    info(f"Categories: {dict(cat_counts)}")
    stats["categories"] = dict(cat_counts)
else:
    fail("metadata.jsonl NOT found!")
    info(f"Searched: {metadata_search}")
    stats["questions"] = 0

# ============================================================
# 9. DATASET: IMAGES (figures/ folder)
# ============================================================
print(f"\n[9/{TOTAL_CHECKS}] Dataset: Images (figures/)")
figures_dir = None
if metadata_path:
    figures_dir = os.path.join(os.path.dirname(metadata_path), "figures")

if figures_dir and os.path.exists(figures_dir):
    case_folders = [d for d in os.listdir(figures_dir) if os.path.isdir(os.path.join(figures_dir, d))]
    total_images = sum(1 for _, _, files in os.walk(figures_dir) for f in files)
    ok(f"{len(case_folders)} case folders, {total_images} image files")
    stats["case_folders"] = len(case_folders)
    stats["total_images"] = total_images

    # Verify images referenced in metadata actually exist
    if metadata_path:
        missing_images = 0
        checked = 0
        sample_missing = []
        base_dir = os.path.dirname(metadata_path)
        with open(metadata_path) as f:
            for line in f:
                entry = json.loads(line)
                for img_path in entry.get("images", []):
                    full_img = os.path.join(base_dir, img_path)
                    checked += 1
                    if not os.path.exists(full_img):
                        missing_images += 1
                        if len(sample_missing) < 3:
                            sample_missing.append(img_path)

        if missing_images == 0:
            ok(f"All {checked} image references resolved to files on disk")
        else:
            fail(f"{missing_images}/{checked} images missing from disk!")
            for s in sample_missing:
                info(f"  Missing: {s}")
            info("Did you unzip figures.zip?")
        stats["images_checked"] = checked
        stats["images_missing"] = missing_images

    # Check formats and corrupt files
    extensions = Counter()
    tiny_images = 0
    for root, dirs, files in os.walk(figures_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            extensions[ext] += 1
            fpath = os.path.join(root, f)
            if os.path.getsize(fpath) < 100:
                tiny_images += 1
    info(f"Formats: {dict(extensions)}")
    if tiny_images > 0:
        warn(f"{tiny_images} images <100 bytes (possibly corrupt)")
    stats["image_formats"] = dict(extensions)
else:
    fail("figures/ folder NOT found!")
    info("Run: huggingface-cli download wanglab/chestagentbench --repo-type dataset --local-dir chestagentbench")
    info("Then unzip: import zipfile; zipfile.ZipFile('chestagentbench/figures.zip').extractall('chestagentbench/')")
    stats["total_images"] = 0
    stats["images_missing"] = "all"

# ============================================================
# 10. SYSTEM PROMPTS
# ============================================================
print(f"\n[10/{TOTAL_CHECKS}] System Prompts")
prompt_search = ["medrax/docs/system_prompts.txt"]
if medrax_root:
    prompt_search.append(os.path.join(medrax_root, "medrax", "docs", "system_prompts.txt"))

prompt_found = False
for pp in prompt_search:
    if os.path.exists(pp):
        with open(pp) as f:
            content = f.read()
        sections = [s for s in content.split("[") if s.strip()]
        ok(f"Found with {len(sections)} prompt section(s)")
        prompt_found = True
        break
if not prompt_found:
    fail("system_prompts.txt NOT found")

# ============================================================
# 11. OPENAI API KEY
# ============================================================
print(f"\n[11/{TOTAL_CHECKS}] OpenAI API Key")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
if api_key:
    masked = api_key[:8] + "..." + api_key[-4:]
    ok(f"OPENAI_API_KEY set: {masked}")
    if base_url:
        info(f"OPENAI_BASE_URL: {base_url}")
else:
    fail("OPENAI_API_KEY not set! Add in Kaggle: Secrets > OPENAI_API_KEY")

# ============================================================
# 12. API CONNECTIVITY TEST
# ============================================================
print(f"\n[12/{TOTAL_CHECKS}] API Connectivity Test")
if api_key:
    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Reply with only: OK"}],
            max_tokens=5,
        )
        reply = response.choices[0].message.content.strip()
        ok(f"GPT-4o responded: '{reply}'")
        stats["api"] = "connected"
    except Exception as e:
        fail(f"API call failed: {e}")
        stats["api"] = "failed"
else:
    warn("Skipped - no API key")
    stats["api"] = "no key"

# ============================================================
# 13. CONFLICT RESOLUTION PIPELINE TEST
# ============================================================
print(f"\n[13/{TOTAL_CHECKS}] Conflict Resolution Pipeline")
if medrax_root and agent_fail == 0:
    try:
        from medrax.agent import (
            CanonicalFinding, ConflictDetector, ConflictResolver,
            ArgumentGraphBuilder, ToolTrustManager, AbstentionLogic
        )

        # Create two contradictory synthetic findings
        finding_a = CanonicalFinding(
            pathology="pneumothorax",
            region="right lung",
            confidence=0.85,
            evidence_type="classification",
            source_tool="ChestXRayClassifier",
            raw_value={"Pneumothorax": 0.85},
            metadata={"details": "Large right-sided pneumothorax detected"}
        )
        finding_b = CanonicalFinding(
            pathology="pneumothorax",
            region="right lung",
            confidence=0.72,
            evidence_type="vqa",
            source_tool="XRayVQA",
            raw_value={"answer": "No pneumothorax identified"},
            metadata={"details": "No pneumothorax identified"}
        )

        # Test conflict detection
        detector = ConflictDetector(sensitivity=0.4)
        conflicts = detector.detect_conflicts([finding_a, finding_b])

        if len(conflicts) > 0:
            ok(f"Conflict detection: found {len(conflicts)} conflict(s) from contradictory inputs")

            # Test conflict resolution
            resolver = ConflictResolver(deferral_threshold=0.6)
            resolution = resolver.resolve_conflict(conflicts[0], [finding_a, finding_b])
            ok(f"Resolution decision: {resolution.get('decision', 'N/A')[:60]}")
            ok(f"Resolution confidence: {resolution.get('confidence', 0):.1%}")
            if resolution.get('should_defer'):
                info("Flagged for human review (normal for close calls)")
            stats["pipeline"] = "fully working"
        else:
            warn("Detector found 0 conflicts on contradictory inputs - check sensitivity")
            stats["pipeline"] = "detection issue"

        # Test premium sub-components
        try:
            builder = ArgumentGraphBuilder()
            ok("ArgumentGraphBuilder initialized")
        except Exception as e:
            warn(f"ArgumentGraphBuilder: {e}")

        try:
            trust_mgr = ToolTrustManager()
            ok("ToolTrustManager initialized")
        except Exception as e:
            warn(f"ToolTrustManager: {e}")

        try:
            abstention = AbstentionLogic()
            ok("AbstentionLogic initialized")
        except Exception as e:
            warn(f"AbstentionLogic: {e}")

    except Exception as e:
        fail(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        stats["pipeline"] = "failed"
else:
    warn("Skipped - agent imports failed earlier")
    stats["pipeline"] = "skipped"

# ============================================================
# 14. FULL AGENT INSTANTIATION TEST
# ============================================================
print(f"\n[14/{TOTAL_CHECKS}] Agent Instantiation Test")
if medrax_root and agent_fail == 0 and api_key:
    try:
        from medrax.agent import Agent
        from medrax.utils import load_prompts_from_file
        from langchain_openai import ChatOpenAI

        prompt_path = None
        for pp in prompt_search:
            if os.path.exists(pp):
                prompt_path = pp
                break

        prompts = load_prompts_from_file(prompt_path) if prompt_path else {}
        system_prompt = prompts.get("MEDICAL_ASSISTANT", "You are a medical AI assistant.")

        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            base_url=base_url if base_url else None,
            max_tokens=1024,
        )

        agent = Agent(
            model=llm,
            tools=[],  # No tools needed - just testing init
            system_prompt=system_prompt,
            enable_conflict_resolution=True,
            conflict_sensitivity=0.4,
            deferral_threshold=0.6,
        )
        ok("Agent created with enable_conflict_resolution=True")
        ok(f"ConflictDetector sensitivity={agent.conflict_detector.sensitivity}")
        ok(f"ConflictResolver deferral_threshold={agent.conflict_resolver.deferral_threshold}")
        stats["agent_init"] = "OK"
    except Exception as e:
        fail(f"Agent init failed: {e}")
        stats["agent_init"] = f"failed: {e}"
else:
    if not api_key:
        warn("Skipped - no API key")
    else:
        warn("Skipped - agent imports failed")
    stats["agent_init"] = "skipped"


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)

print(f"\n  Checks passed:  {TOTAL_CHECKS - len(issues)}/{TOTAL_CHECKS}")
print(f"  Issues:         {len(issues)}")
print(f"  Warnings:       {len(warnings)}")

if stats.get("questions"):
    print(f"\n  Dataset:        {stats.get('questions', '?')} questions")
    print(f"  Images:         {stats.get('total_images', '?')} files in {stats.get('case_folders', '?')} cases")
    missing = stats.get('images_missing', 0)
    if isinstance(missing, int) and missing > 0:
        print(f"  Missing images: {missing}")
    elif missing == 0:
        print(f"  Missing images: 0 (all present)")

print(f"\n  Packages:       {stats.get('packages', '?')}")
print(f"  Agent imports:  {stats.get('agent_imports', '?')}")
print(f"  Tools imports:  {stats.get('tools_imports', '?')}")
print(f"  Pipeline test:  {stats.get('pipeline', '?')}")
print(f"  API:            {stats.get('api', '?')}")
print(f"  Agent init:     {stats.get('agent_init', '?')}")

if issues:
    print(f"\n  {'='*60}")
    print(f"  ❌ {len(issues)} ISSUE(S) TO FIX:")
    print(f"  {'='*60}")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

if warnings:
    print(f"\n  WARNINGS:")
    for w in warnings:
        print(f"  - {w}")

if not issues:
    print(f"\n  ✅ ALL {TOTAL_CHECKS} CHECKS PASSED!")
    print(f"  READY TO RUN MEDRAX PREMIUM BENCHMARK!")
elif len(issues) == 1 and "tools" in issues[0].lower():
    print(f"\n  ⚠️  Almost ready - check transformers fix (see check 5 above)")
else:
    print(f"\n  ❌ Fix the issues above before running the benchmark.")

print("\n" + "=" * 70)
