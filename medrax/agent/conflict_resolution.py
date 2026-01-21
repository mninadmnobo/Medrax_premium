"""
Conflict detection and resolution for MedRAX tool outputs.
Implements explicit conflict detection with rule-based and probabilistic approaches.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from .canonical_output import CanonicalFinding
from .anatomical_consistency_graph import GACLConflictDetector


@dataclass
class Conflict:
    """
    Represents a detected conflict between tool outputs.
    
    Attributes:
        conflict_type: Type of conflict ("presence", "location", "severity", "value")
        finding: What finding is in conflict (e.g., "Pneumothorax")
        tools_involved: List of tool names that disagree
        values: Conflicting values from each tool
        confidences: Confidence scores from each tool
        severity: How critical is this conflict ("critical", "moderate", "minor")
        recommendation: Suggested resolution approach
        timestamp: When conflict was detected
    """
    conflict_type: str
    finding: str
    tools_involved: List[str]
    values: List[Any]
    confidences: List[float]
    severity: str
    recommendation: str
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        values_str = " vs ".join([f"{t}={v} ({c:.0%})" 
                                   for t, v, c in zip(self.tools_involved, self.values, self.confidences)])
        return f"[{self.severity.upper()}] {self.conflict_type} conflict on '{self.finding}': {values_str}"


class ConflictDetector:
    """
    Detects conflicts in multi-tool outputs using rule-based and probabilistic methods.
    """
    
    # Conflict detection thresholds
    PRESENCE_THRESHOLD_HIGH = 0.7  # >70% = present
    PRESENCE_THRESHOLD_LOW = 0.3   # <30% = absent
    CONFIDENCE_GAP_THRESHOLD = 0.4  # Difference that triggers conflict
    
    # Standard pathology list
    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Fracture", "Hernia", "Infiltration",
        "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
        "Support Devices"
    ]
    
    def __init__(self, sensitivity: float = 0.4):
        """
        Initialize conflict detector.
        
        Args:
            sensitivity: Confidence gap threshold (0-1). Lower = more sensitive.
        """
        self.sensitivity = sensitivity
        self.CONFIDENCE_GAP_THRESHOLD = sensitivity
        
        # Initialize GACL for semantic conflict detection
        self.gacl_detector = GACLConflictDetector(anomaly_threshold=0.6)
    
    def detect_conflicts(self, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Analyze findings for conflicts using 6-step framework:
        1. Output Normalization ✅ (done in canonical_output.py)
        2. Confidence Calibration ✅ (done in canonical_output.py)
        3. Conflict Detection (THIS METHOD)
        4. Task-Aware Arbitration (done in ConflictResolver)
        5. Meta-Reasoner (LLM in main agent loop)
        6. Uncertainty-Aware Deferral (done in ConflictResolver)
        
        For CXR analysis, we detect:
        - Presence/Absence conflicts: Tools disagree on whether finding exists
        - Semantic conflicts: Contradictory interpretations (via GACL for ALL pathologies)
        - Complementary patterns: Co-occurring findings that suggest common pathology
        
        Args:
            findings: List of canonical findings from all tools (already normalized & calibrated)
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group findings by pathology
        by_pathology = self._group_by_pathology(findings)
        
        # Check each pathology for conflicts
        for pathology, tool_findings in by_pathology.items():
            # Need at least 2 tools to have a conflict
            if len(tool_findings) < 2:
                continue
            
            # STEP 3a: Presence/Absence Conflicts (Rule-based)
            # Detects when tools disagree on whether a CXR finding is present
            presence_conflicts = self._check_presence_conflicts(pathology, tool_findings)
            conflicts.extend(presence_conflicts)
            
            # STEP 3b: Semantic/Contradictory Conflicts (Probabilistic via GACL)
            # Uses Graph-Based Anatomical Consistency Learning for all CXR findings
            # Detects when segmentation patterns contradict classification predictions
            semantic_conflicts = self._check_semantic_conflicts_gacl(pathology, tool_findings)
            conflicts.extend(semantic_conflicts)
        
        return conflicts
    
    def _group_by_pathology(self, findings: List[CanonicalFinding]) -> Dict[str, List[CanonicalFinding]]:
        """Group findings by pathology name."""
        grouped = {}
        for finding in findings:
            if finding.pathology not in grouped:
                grouped[finding.pathology] = []
            grouped[finding.pathology].append(finding)
        return grouped
    
    def _check_presence_conflicts(self, pathology: str, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Check if tools disagree on whether a finding is present or absent.
        
        Rules:
        - >0.7 confidence = present
        - <0.3 confidence = absent
        - 0.3-0.7 = uncertain
        
        Conflict if: max_confidence - min_confidence > threshold
        """
        conflicts = []
        
        confidences = [f.confidence for f in findings]
        max_conf = max(confidences)
        min_conf = min(confidences)
        
        # Check for significant disagreement
        if max_conf - min_conf > self.CONFIDENCE_GAP_THRESHOLD:
            # One tool says present, another says absent
            if max_conf > self.PRESENCE_THRESHOLD_HIGH and min_conf < self.PRESENCE_THRESHOLD_LOW:
                severity = "critical" if max_conf > 0.85 else "moderate"
                
                conflict = Conflict(
                    conflict_type="presence",
                    finding=pathology,
                    tools_involved=[f.source_tool for f in findings],
                    values=["present" if f.confidence > 0.5 else "absent" for f in findings],
                    confidences=confidences,
                    severity=severity,
                    recommendation=self._get_presence_resolution_strategy(findings)
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _get_presence_resolution_strategy(self, findings: List[CanonicalFinding]) -> str:
        """Recommend how to resolve presence/absence conflict."""
        # Find highest confidence finding
        max_finding = max(findings, key=lambda f: f.confidence)
        
        if max_finding.confidence > 0.85:
            return f"High confidence from {max_finding.source_tool} ({max_finding.confidence:.0%}). Recommend: Trust primary tool, but flag for review."
        elif max_finding.confidence > 0.7:
            return f"Moderate confidence from {max_finding.source_tool} ({max_finding.confidence:.0%}). Recommend: Call additional verification tool."
        else:
            return "Low confidence across all tools. Recommend: Defer to radiologist review."
    
    def _check_semantic_conflicts_gacl(self, pathology: str, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Detect semantic conflicts using GACL for ALL CXR pathologies.
        
        GACL (Graph-Based Anatomical Consistency Learning) detects when:
        - Segmentation finds anatomical patterns inconsistent with classification
        - Multiple tools contradict each other on disease presence
        - Anatomical measurements suggest pathology not detected by classifier
        
        Works for ALL pathologies: Pneumonia, Pneumothorax, Cardiomegaly, Effusion, etc.
        
        Args:
            pathology: The finding being checked (e.g., "Pneumonia", "Cardiomegaly")
            findings: Findings from different tools for this pathology
            
        Returns:
            List of conflicts (0 or 1) if semantic conflict detected
        """
        # Find segmentation and classification pairs
        seg_findings = [f for f in findings if "segmentation" in f.source_tool.lower()]
        clf_findings = [f for f in findings if "classifier" in f.source_tool.lower()]
        
        # Need both types to detect semantic conflict
        if not (seg_findings and clf_findings):
            return []
        
        # Use GACL for all pathologies
        try:
            for seg in seg_findings:
                for clf in clf_findings:
                    conflict = self.detect_semantic_conflict_gacl(
                        seg.raw_value,  # Segmentation measurements
                        clf.raw_value   # Classification prediction
                    )
                    if conflict:
                        return [conflict]
        except Exception as e:
            # Gracefully handle GACL failures
            print(f"  ⚠️ GACL analysis failed for {pathology}: {str(e)}")
        
        return []
    
    def _check_complementary_diseases(self, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Step 3d: Detect complementary disease patterns.
        
        Examples of complementary findings:
        - Pneumonia + Pleural Effusion (common together)
        - Cardiomegaly + Pulmonary Edema (heart failure signs)
        - Consolidation + Atelectasis (same general pathology area)
        
        This is NOT a conflict - it's a pattern recognition step.
        Returns empty list (no conflict) but logs for clinical correlation.
        """
        # Currently, complementary findings are reported together, not as conflicts
        # In future, could flag as "high-risk pattern" if many co-occur
        return []
    
    def detect_semantic_conflict_gacl(self, seg_output: Dict[str, Any], 
                                      classifier_output: Dict[str, Any]) -> Optional[Conflict]:
        """
        Detect semantic conflicts using Graph-Based Anatomical Consistency Learning.
        
        Detects when segmentation finds anatomical/structural anomalies but classifier 
        predicts normal (or vice versa). Uses learned anatomical patterns rather than 
        hardcoded rules.
        
        Works for ALL CXR pathologies:
        - Cardiac: Cardiomegaly, Enlarged Cardiomediastinum
        - Lung: Pneumonia, Pneumothorax, Consolidation
        - Pleural: Effusion, Pleural Thickening
        - Others: Nodule, Mass, Fracture, etc.
        
        Args:
            seg_output: Segmentation tool output with anatomical measurements
            classifier_output: Classification tool output
        
        Returns:
            Conflict object if semantic conflict detected, None otherwise
        """
        try:
            # Run GACL analysis
            analysis = self.gacl_detector.detect_semantic_conflict(seg_output, classifier_output)
            
            if not analysis.get("has_conflict", False):
                return None
            
            # Create conflict representation (generic for all pathologies)
            conflict = Conflict(
                conflict_type="semantic",
                finding="Anatomical pattern consistency",
                tools_involved=["segmentation_tool", "classification_model"],
                values=[
                    analysis["segmentation_pattern"],
                    analysis["classification_prediction"]
                ],
                confidences=[
                    1.0 - analysis["anomaly_score"],  # Abnormality confidence
                    analysis["classification_confidence"]
                ],
                severity="critical",
                recommendation=(
                    f"Graph-based anatomical analysis detected inconsistency. "
                    f"Segmentation analysis suggests: {analysis['most_likely_diagnosis']} "
                    f"({analysis['confidence_in_diagnosis']:.0%} confidence). "
                    f"Reason: {analysis['explanation']}"
                )
            )
            
            return conflict
        
        except Exception:
            # Gracefully handle GACL analysis failures
            return None


class ConflictResolver:
    """
    Resolves conflicts using task-aware arbitration and tool expertise hierarchy.
    """
    
    # Tool expertise hierarchy: which tool is best for which task
    TOOL_EXPERTISE = {
        "presence_detection": {
            "primary": "chest_xray_classifier",
            "fallback": "chest_xray_expert",
            "description": "Binary presence/absence of pathology"
        },
        "localization": {
            "primary": "segmentation_tool",
            "fallback": "phrase_grounding_tool",
            "description": "Precise anatomical location"
        },
        "description": {
            "primary": "chest_xray_expert",
            "fallback": "chest_xray_report_generator",
            "description": "Detailed clinical description"
        },
        "severity_assessment": {
            "primary": "chest_xray_expert",
            "fallback": "chest_xray_classifier",
            "description": "Severity grading (mild/moderate/severe)"
        }
    }
    
    def __init__(self, deferral_threshold: float = 0.6):
        """
        Initialize conflict resolver.
        
        Args:
            deferral_threshold: Confidence below which we defer to humans
        """
        self.deferral_threshold = deferral_threshold
    
    def resolve_conflict(self, conflict: Conflict, findings: List[CanonicalFinding]) -> Dict[str, Any]:
        """
        Resolve a conflict using task-aware arbitration.
        
        Args:
            conflict: Detected conflict
            findings: All findings related to this conflict
            
        Returns:
            Resolution dict with decision, confidence, and reasoning
        """
        # Determine task type from conflict type
        if conflict.conflict_type == "presence":
            task_type = "presence_detection"
        elif conflict.conflict_type == "location":
            task_type = "localization"
        elif conflict.conflict_type == "severity":
            task_type = "severity_assessment"
        else:
            task_type = "description"
        
        # Get expert tool for this task
        expertise = self.TOOL_EXPERTISE.get(task_type, {})
        primary_tool = expertise.get("primary", None)
        
        # Find primary tool's finding
        primary_finding = None
        for finding in findings:
            if finding.source_tool == primary_tool:
                primary_finding = finding
                break
        
        # If primary tool available, trust it
        if primary_finding:
            resolution = {
                "decision": "trust_primary_tool",
                "selected_tool": primary_tool,
                "value": primary_finding.confidence > 0.5,  # Present or absent
                "confidence": primary_finding.confidence,
                "reasoning": f"Primary tool for {task_type} is {primary_tool}",
                "should_defer": primary_finding.confidence < self.deferral_threshold
            }
        else:
            # No primary tool, use weighted average
            resolution = self._weighted_average_resolution(findings)
        
        return resolution
    
    def _weighted_average_resolution(self, findings: List[CanonicalFinding]) -> Dict[str, Any]:
        """
        Resolve conflict using confidence-weighted average.
        
        Args:
            findings: All findings for this pathology
            
        Returns:
            Resolution dict
        """
        # Weight by confidence
        total_weight = sum(f.confidence for f in findings)
        
        if total_weight == 0:
            return {
                "decision": "insufficient_confidence",
                "confidence": 0.0,
                "reasoning": "All tools have zero confidence",
                "should_defer": True
            }
        
        # Weighted average of "presence" (1 if confidence > 0.5, else 0)
        weighted_presence = sum(
            f.confidence * (1 if f.confidence > 0.5 else 0)
            for f in findings
        ) / total_weight
        
        avg_confidence = sum(f.confidence for f in findings) / len(findings)
        
        return {
            "decision": "weighted_average",
            "value": weighted_presence > 0.5,
            "confidence": avg_confidence,
            "reasoning": f"Averaged {len(findings)} tool outputs with confidence weighting",
            "should_defer": avg_confidence < self.deferral_threshold
        }
    
    def should_defer_to_human(self, resolution: Dict[str, Any]) -> bool:
        """
        Decide if this case should be deferred to human review.
        
        Args:
            resolution: Resolution decision
            
        Returns:
            True if should defer to radiologist
        """
        return resolution.get("should_defer", False)


def generate_conflict_report(conflicts: List[Conflict], resolutions: List[Dict[str, Any]] = None) -> str:
    """
    Generate human-readable conflict report.
    
    Args:
        conflicts: List of detected conflicts
        resolutions: Optional list of resolutions (aligned with conflicts)
        
    Returns:
        Formatted report string
    """
    if not conflicts:
        return "✅ No conflicts detected - all tools agree"
    
    report = "⚠️  CONFLICT DETECTION REPORT\n"
    report += "=" * 60 + "\n"
    report += f"Detected {len(conflicts)} conflict(s)\n"
    report += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, conflict in enumerate(conflicts, 1):
        report += f"Conflict #{i} - {conflict.severity.upper()} SEVERITY\n"
        report += "-" * 60 + "\n"
        report += f"Type: {conflict.conflict_type}\n"
        report += f"Finding: {conflict.finding}\n"
        report += f"Tools: {', '.join(conflict.tools_involved)}\n"
        
        # Show disagreement details
        for tool, value, conf in zip(conflict.tools_involved, conflict.values, conflict.confidences):
            report += f"  • {tool}: {value} (confidence: {conf:.1%})\n"
        
        report += f"Recommendation: {conflict.recommendation}\n"
        
        # Add resolution if available
        if resolutions and i <= len(resolutions):
            res = resolutions[i-1]
            report += "\nResolution:\n"
            report += f"  Decision: {res.get('decision', 'N/A')}\n"
            report += f"  Selected: {res.get('selected_tool', 'N/A')}\n"
            report += f"  Confidence: {res.get('confidence', 0):.1%}\n"
            report += f"  Reasoning: {res.get('reasoning', 'N/A')}\n"
            if res.get('should_defer', False):
                report += "  ⚠️  FLAGGED FOR HUMAN REVIEW\n"
        
        report += "\n"
    
    return report
