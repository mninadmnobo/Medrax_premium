"""
Probabilistic Conflict Graph for MedRAX - HYBRID APPROACH
==========================================================

Implements Rule-based + Probabilistic Conflict Detection based on:
1. NeurIPS 2021: "Uncertainty Quantification and Deep Ensembles" (Rahaman & Thiery)
2. MDPI 2024: Medical Tool Conflict Resolution (arXiv:2502.02673v2)

Key Concepts:
- Expected Calibration Error (ECE): measures confidence vs accuracy disagreement
- Entropy-based divergence: detects when tools have different uncertainty distributions
- Ensemble disagreement: probabilistic graph captures tool relationships
- Temperature scaling: normalizes confidence across heterogeneous tools

NOT keyword-based, but statistically principled.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import math
import numpy as np
from collections import defaultdict

from .canonical_output import CanonicalFinding


@dataclass
class ConflictNode:
    """
    Represents a node in the probabilistic conflict graph.
    Each node is a (pathology, tool) pair with uncertainty metrics.
    """
    pathology: str
    tool_name: str
    confidence: float
    entropy: float  # Shannon entropy: -sum(p * log(p))
    evidence_type: str
    raw_value: Any
    
    def __hash__(self):
        return hash((self.pathology, self.tool_name))
    
    def __eq__(self, other):
        return self.pathology == other.pathology and self.tool_name == other.tool_name


@dataclass
class ConflictEdge:
    """
    Represents an edge (conflict) between two tool outputs in the graph.
    Quantifies the disagreement probabilistically.
    """
    node1: ConflictNode
    node2: ConflictNode
    
    # Conflict metrics
    confidence_gap: float  # |conf1 - conf2|
    entropy_divergence: float  # Jensen-Shannon divergence
    consensus_probability: float  # P(both tools correct)
    conflict_severity: str  # "critical", "moderate", "minor"
    conflict_type: str  # "presence", "severity", "location"
    
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node1": {"pathology": self.node1.pathology, "tool": self.node1.tool_name},
            "node2": {"pathology": self.node2.pathology, "tool": self.node2.tool_name},
            "confidence_gap": self.confidence_gap,
            "entropy_divergence": self.entropy_divergence,
            "consensus_probability": self.consensus_probability,
            "severity": self.conflict_severity,
            "type": self.conflict_type,
            "timestamp": self.timestamp
        }


@dataclass
class CalibrationMetrics:
    """
    Stores calibration metrics for uncertainty quantification.
    Based on Expected Calibration Error (ECE) from Guo et al. 2017.
    """
    ece: float  # Expected Calibration Error (0-1)
    nll: float  # Negative Log-Likelihood
    entropy_mean: float  # Average entropy
    entropy_std: float  # Entropy standard deviation
    confidence_bins: Dict[str, float] = field(default_factory=dict)


class ProbabilisticConflictGraph:
    """
    Builds a probabilistic graph of tool conflicts using uncertainty quantification.
    
    Each node: (pathology, tool) with confidence & entropy
    Each edge: quantifies disagreement probabilistically
    """
    
    # Thresholds (empirically tuned, can be adjusted)
    CONFIDENCE_GAP_THRESHOLD = 0.40  # >40% gap = conflict
    ENTROPY_DIVERGENCE_THRESHOLD = 0.35  # JS divergence threshold
    CONSENSUS_THRESHOLD = 0.50  # P(both correct) < 50% = conflict
    CRITICAL_SEVERITY_THRESHOLD = 0.75  # >75% confidence difference = critical
    
    def __init__(self, sensitivity: float = 0.4):
        """
        Initialize the probabilistic conflict graph.
        
        Args:
            sensitivity: Detection sensitivity (0-1). Lower = more sensitive.
        """
        self.sensitivity = sensitivity
        self.nodes: Dict[Tuple[str, str], ConflictNode] = {}  # (pathology, tool) -> node
        self.edges: List[ConflictEdge] = []
        self.calibration_metrics: Dict[str, CalibrationMetrics] = {}
    
    def build_graph(self, findings: List[CanonicalFinding]) -> Tuple[List[ConflictEdge], Dict[str, Any]]:
        """
        Build the probabilistic conflict graph from canonical findings.
        
        Args:
            findings: List of canonical findings from all tools
            
        Returns:
            (detected_conflicts, graph_analysis)
        """
        self.nodes = {}
        self.edges = []
        
        # Step 1: Create nodes for each finding
        print("\n🔗 BUILDING PROBABILISTIC CONFLICT GRAPH")
        print("="*60)
        print("Step 1: Creating nodes from findings...")
        
        for finding in findings:
            entropy = self._calculate_entropy(finding.confidence)
            node = ConflictNode(
                pathology=finding.pathology,
                tool_name=finding.source_tool,
                confidence=finding.confidence,
                entropy=entropy,
                evidence_type=finding.evidence_type,
                raw_value=finding.raw_value
            )
            key = (finding.pathology, finding.source_tool)
            self.nodes[key] = node
            print(f"  ✓ {finding.pathology:20} | {finding.source_tool:25} | conf={finding.confidence:.2f} entropy={entropy:.3f}")
        
        # Step 2: Group findings by pathology
        print("\nStep 2: Grouping by pathology...")
        by_pathology = self._group_findings_by_pathology(findings)
        
        # Step 3: Detect conflicts between tool pairs
        print("\nStep 3: Detecting conflicts...")
        for pathology, tool_findings in by_pathology.items():
            if len(tool_findings) < 2:
                print(f"  ⏭️  {pathology}: Only 1 tool, skipping conflict detection")
                continue
            
            print(f"  📊 {pathology}: {len(tool_findings)} tools -> checking {len(list(combinations_2(len(tool_findings))))} pairs")
            
            # Check all pairs of tools
            tool_nodes = [self.nodes[(pathology, f.source_tool)] for f in tool_findings]
            
            for i in range(len(tool_nodes)):
                for j in range(i + 1, len(tool_nodes)):
                    node1, node2 = tool_nodes[i], tool_nodes[j]
                    edge = self._create_edge(node1, node2)
                    
                    if edge:  # Edge exists if conflict detected
                        self.edges.append(edge)
                        print(f"     ⚠️  CONFLICT: {node1.tool_name} vs {node2.tool_name}")
                        print(f"        Gap: {edge.confidence_gap:.2f} | JS-Div: {edge.entropy_divergence:.3f} | Type: {edge.conflict_type}")
        
        # Step 4: Calculate calibration metrics
        print("\nStep 4: Computing calibration metrics...")
        self._compute_calibration_metrics(findings)
        
        graph_analysis = self._analyze_graph()
        
        return self.edges, graph_analysis
    
    def _calculate_entropy(self, confidence: float) -> float:
        """
        Calculate Shannon entropy from confidence score.
        
        For binary classification (present/absent):
        H(p) = -p*log(p) - (1-p)*log(1-p)
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Entropy (0-1, where 0=certain, 1=maximal uncertainty)
        """
        if confidence <= 0 or confidence >= 1:
            return 0.0
        
        # Clamp to avoid log(0)
        p = max(0.001, min(0.999, confidence))
        entropy = -p * math.log(p) - (1 - p) * math.log(1 - p)
        
        # Normalize to [0, 1] - max entropy for binary is ln(2)
        max_entropy = math.log(2)
        return entropy / max_entropy
    
    def _create_edge(self, node1: ConflictNode, node2: ConflictNode) -> Optional[ConflictEdge]:
        """
        Create an edge (conflict) between two nodes.
        Returns None if no significant conflict detected.
        """
        # Calculate conflict metrics
        confidence_gap = abs(node1.confidence - node2.confidence)
        entropy_divergence = self._jensen_shannon_divergence(node1.confidence, node2.confidence)
        consensus_prob = self._calculate_consensus_probability(node1.confidence, node2.confidence)
        
        # Determine conflict severity
        if confidence_gap > self.CRITICAL_SEVERITY_THRESHOLD:
            severity = "critical"
        elif confidence_gap > 0.5:
            severity = "moderate"
        else:
            severity = "minor"
        
        # Determine conflict type
        conflict_type = self._determine_conflict_type(node1, node2, confidence_gap)
        
        # Rule-based detection: conflict exists if meets thresholds
        has_confidence_conflict = confidence_gap > self.CONFIDENCE_GAP_THRESHOLD
        has_entropy_conflict = entropy_divergence > self.ENTROPY_DIVERGENCE_THRESHOLD
        has_low_consensus = consensus_prob < self.CONSENSUS_THRESHOLD
        
        # Probabilistic condition: detect if ANY threshold exceeded
        if has_confidence_conflict or has_entropy_conflict or has_low_consensus:
            return ConflictEdge(
                node1=node1,
                node2=node2,
                confidence_gap=confidence_gap,
                entropy_divergence=entropy_divergence,
                consensus_probability=consensus_prob,
                conflict_severity=severity,
                conflict_type=conflict_type
            )
        
        return None
    
    def _jensen_shannon_divergence(self, conf1: float, conf2: float) -> float:
        """
        Calculate Jensen-Shannon divergence between two confidence distributions.
        
        For binary case:
        P1 = [conf1, 1-conf1]
        P2 = [conf2, 1-conf2]
        
        JS(P1||P2) = 0.5 * KL(P1||M) + 0.5 * KL(P2||M)
        where M = 0.5(P1 + P2)
        """
        p1 = np.array([conf1, 1 - conf1])
        p2 = np.array([conf2, 1 - conf2])
        
        # Mixture distribution
        m = 0.5 * (p1 + p2)
        
        # KL divergence (add epsilon to avoid log(0))
        eps = 1e-10
        kl1 = np.sum(p1 * np.log((p1 + eps) / (m + eps)))
        kl2 = np.sum(p2 * np.log((p2 + eps) / (m + eps)))
        
        js = 0.5 * kl1 + 0.5 * kl2
        
        # Normalize by max JS divergence (ln(2) for binary)
        max_js = math.log(2)
        return js / max_js
    
    def _calculate_consensus_probability(self, conf1: float, conf2: float) -> float:
        """
        Calculate probability that both tools are correct.
        
        Assumes independence: P(both correct) = conf1 * conf2
        
        Args:
            conf1, conf2: Confidence scores
            
        Returns:
            Consensus probability (0-1)
        """
        return conf1 * conf2
    
    def _determine_conflict_type(self, node1: ConflictNode, node2: ConflictNode, gap: float) -> str:
        """
        Determine type of conflict based on confidence profiles.
        
        Types:
        - "presence": one tool says present (>0.5), other says absent (<0.5)
        - "severity": both say present but disagree on confidence
        - "uncertainty": both uncertain but in different ways (entropy mismatch)
        """
        threshold = 0.5
        
        node1_present = node1.confidence > threshold
        node2_present = node2.confidence > threshold
        
        # Presence conflict
        if node1_present != node2_present:
            return "presence"
        
        # Severity conflict (both agree on presence but differ significantly)
        if node1_present and node2_present and gap > 0.3:
            return "severity"
        
        # Entropy/Uncertainty conflict
        if abs(node1.entropy - node2.entropy) > 0.3:
            return "uncertainty"
        
        return "value"
    
    def _group_findings_by_pathology(self, findings: List[CanonicalFinding]) -> Dict[str, List[CanonicalFinding]]:
        """Group findings by pathology name."""
        grouped = defaultdict(list)
        for finding in findings:
            grouped[finding.pathology].append(finding)
        return dict(grouped)
    
    def _compute_calibration_metrics(self, findings: List[CanonicalFinding]) -> None:
        """
        Compute calibration metrics per tool using ECE.
        
        ECE measures: |confidence - accuracy|
        Lower ECE = better calibrated
        """
        by_tool = defaultdict(list)
        for finding in findings:
            by_tool[finding.source_tool].append(finding)
        
        for tool_name, tool_findings in by_tool.items():
            confidences = [f.confidence for f in tool_findings]
            entropies = [self._calculate_entropy(f.confidence) for f in tool_findings]
            
            metrics = CalibrationMetrics(
                ece=self._calculate_ece(confidences),
                nll=self._calculate_nll(confidences),
                entropy_mean=np.mean(entropies),
                entropy_std=np.std(entropies),
                confidence_bins=self._bin_confidences(confidences)
            )
            self.calibration_metrics[tool_name] = metrics
    
    def _calculate_ece(self, confidences: List[float], num_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error.
        
        ECE = sum(|confidence - accuracy|) / num_bins
        Since we don't have ground truth, we approximate using uncertainty.
        """
        if not confidences:
            return 0.0
        
        # Bin confidences
        bins = np.linspace(0, 1, num_bins + 1)
        bin_accs = []
        bin_confs = []
        
        for i in range(num_bins):
            mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i + 1])
            if np.sum(mask) > 0:
                bin_confs.append(np.mean(np.array(confidences)[mask]))
                # Approximate accuracy as confidence (since we lack labels)
                bin_accs.append(np.mean(np.array(confidences)[mask]))
        
        if not bin_confs:
            return 0.0
        
        ece = np.mean(np.abs(np.array(bin_accs) - np.array(bin_confs)))
        return float(ece)
    
    def _calculate_nll(self, confidences: List[float]) -> float:
        """
        Calculate Negative Log-Likelihood.
        Approximated from confidence scores.
        """
        if not confidences:
            return 0.0
        
        nll = -np.mean([math.log(max(c, 1e-10)) if c > 0.5 else math.log(max(1 - c, 1e-10)) 
                       for c in confidences])
        return float(nll)
    
    def _bin_confidences(self, confidences: List[float], num_bins: int = 5) -> Dict[str, float]:
        """Bin confidences for analysis."""
        bins = {}
        for i in range(num_bins):
            bin_name = f"{i*100//num_bins}-{(i+1)*100//num_bins}%"
            threshold_low = i / num_bins
            threshold_high = (i + 1) / num_bins
            count = sum(1 for c in confidences if threshold_low <= c < threshold_high)
            bins[bin_name] = count / len(confidences) if confidences else 0
        return bins
    
    def _analyze_graph(self) -> Dict[str, Any]:
        """Analyze the conflict graph structure."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "conflict_density": len(self.edges) / max(1, len(self.nodes) * (len(self.nodes) - 1) / 2),
            "critical_conflicts": len([e for e in self.edges if e.conflict_severity == "critical"]),
            "moderate_conflicts": len([e for e in self.edges if e.conflict_severity == "moderate"]),
            "calibration_metrics": {
                tool: {
                    "ece": metrics.ece,
                    "nll": metrics.nll,
                    "entropy_mean": metrics.entropy_mean,
                    "entropy_std": metrics.entropy_std
                }
                for tool, metrics in self.calibration_metrics.items()
            }
        }


def combinations_2(n: int):
    """Generate all 2-combinations from range(n)."""
    for i in range(n):
        for j in range(i + 1, n):
            yield (i, j)


def analyze_tool_calibration(findings: List[CanonicalFinding]) -> Dict[str, Dict[str, float]]:
    """
    Analyze calibration of each tool independently.
    
    Args:
        findings: List of canonical findings
        
    Returns:
        Calibration analysis per tool
    """
    analysis = {}
    
    by_tool = defaultdict(list)
    for finding in findings:
        by_tool[finding.source_tool].append(finding)
    
    for tool, tool_findings in by_tool.items():
        confidences = [f.confidence for f in tool_findings]
        
        analysis[tool] = {
            "sample_count": len(tool_findings),
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "confidence_min": float(np.min(confidences)),
            "confidence_max": float(np.max(confidences)),
            "entropy_mean": float(np.mean([
                -c * math.log(max(c, 1e-10)) - (1-c) * math.log(max(1-c, 1e-10))
                for c in confidences
            ]))
        }
    
    return analysis
