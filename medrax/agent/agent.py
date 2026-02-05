import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .canonical_output import normalize_output, CanonicalFinding
from .conflict_resolution import ConflictDetector, ConflictResolver, generate_conflict_report
from .probabilistic_conflict_graph import ProbabilisticConflictGraph, analyze_tool_calibration

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        enable_conflict_resolution: bool = True,
        conflict_sensitivity: float = 0.4,
        deferral_threshold: float = 0.6,
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
            enable_conflict_resolution (bool, optional): Enable conflict detection/resolution. Defaults to True.
            conflict_sensitivity (float, optional): Conflict detection sensitivity (0-1). Defaults to 0.4.
            deferral_threshold (float, optional): Confidence threshold for human deferral. Defaults to 0.6.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        self.enable_conflict_resolution = enable_conflict_resolution

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)
        
        # Initialize conflict detection and resolution
        if self.enable_conflict_resolution:
            self.conflict_detector = ConflictDetector(sensitivity=conflict_sensitivity)
            self.conflict_resolver = ConflictResolver(deferral_threshold=deferral_threshold)
            self.probabilistic_graph = ProbabilisticConflictGraph(sensitivity=conflict_sensitivity)
            print(f"✅ Conflict resolution enabled (sensitivity={conflict_sensitivity}, deferral={deferral_threshold})")
        
        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response with conflict detection and resolution.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []
        canonical_findings = []
        
        # STEP 1: Execute all tools
        print("\n" + "="*60)
        print(f"🔧 Executing {len(tool_calls)} tool(s)")
        print("="*60)

        for i, call in enumerate(tool_calls, 1):
            print(f"\n[{i}/{len(tool_calls)}] Tool: {call['name']}")
            
            if call["name"] not in self.tools:
                print("  ❌ Invalid tool")
                result = "invalid tool, please retry"
            else:
                try:
                    result = self.tools[call["name"]].invoke(call["args"])
                    print("  ✅ Execution successful")
                    
                    # STEP 2: Normalize output to canonical format (if conflict resolution enabled)
                    if self.enable_conflict_resolution:
                        tool_type = self._get_tool_type(call["name"])
                        normalized = normalize_output(
                            result, 
                            call["name"], 
                            tool_type,
                            **call.get("args", {})
                        )
                        canonical_findings.extend(normalized)
                        print(f"  📊 Normalized to {len(normalized)} finding(s)")
                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    result = f"Error executing tool: {str(e)}"

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )
        
        # STEP 3: Conflict Detection (if enabled and we have multiple findings)
        conflicts = []
        resolutions = []
        
        if self.enable_conflict_resolution and len(canonical_findings) > 1:
            print(f"\n{'='*60}")
            print("🔍 CONFLICT DETECTION")
            print("="*60)
            print(f"Analyzing {len(canonical_findings)} canonical findings...")
            
            conflicts = self.conflict_detector.detect_conflicts(canonical_findings)
            
            if conflicts:
                print(f"\n⚠️  Detected {len(conflicts)} conflict(s)!")
                
                # STEP 4: Resolve conflicts
                print("\n" + "="*60)
                print("🔧 CONFLICT RESOLUTION")
                print("="*60)
                
                for i, conflict in enumerate(conflicts, 1):
                    print(f"\n[{i}/{len(conflicts)}] {conflict.to_summary()}")
                    
                    # Get relevant findings for this conflict
                    relevant_findings = [
                        f for f in canonical_findings 
                        if f.pathology == conflict.finding
                    ]
                    
                    resolution = self.conflict_resolver.resolve_conflict(conflict, relevant_findings)
                    resolutions.append(resolution)
                    
                    print(f"  Resolution: {resolution['decision']}")
                    print(f"  Confidence: {resolution['confidence']:.1%}")
                    if resolution.get('should_defer', False):
                        print("  ⚠️  FLAGGED FOR HUMAN REVIEW")
                
                # Generate conflict report
                conflict_report = generate_conflict_report(conflicts, resolutions)
                
                # Add conflict report as a special tool message
                conflict_message = ToolMessage(
                    tool_call_id="conflict_analysis",
                    name="ConflictAnalyzer",
                    args={},
                    content=conflict_report
                )
                results.append(conflict_message)
                print(f"\n📋 Conflict report added to results")
            else:
                print("✅ No conflicts detected - all tools agree")
        
        # STEP 5: Save comprehensive logs
        if self.enable_conflict_resolution:
            self._save_tool_calls_with_conflicts(results, canonical_findings, conflicts, resolutions)
        else:
            self._save_tool_calls(results)
        
        print(f"\n{'='*60}")  
        print("✅ Tool execution complete")
        print("="*60 + "\n")
        print("Returning to model processing!")

        return {"messages": results}
    
    def _get_tool_type(self, tool_name: str) -> str:
        """Determine tool type from tool name."""
        if "classifier" in tool_name.lower() or "classification" in tool_name.lower():
            return "classification"
        elif "vqa" in tool_name.lower() or "expert" in tool_name.lower() or "llava" in tool_name.lower():
            return "vqa"
        elif "segmentation" in tool_name.lower():
            return "segmentation"
        elif "grounding" in tool_name.lower():
            return "grounding"
        elif "report" in tool_name.lower():
            return "report"
        else:
            return "unknown"
    
    def _detect_conflicts_with_probabilistic_graph(self, canonical_findings: List[CanonicalFinding]) -> Tuple[List, Dict]:
        """
        Use probabilistic conflict graph for sophisticated conflict detection.
        
        Based on uncertainty quantification and ensemble disagreement methods.
        """
        graph_edges, graph_analysis = self.probabilistic_graph.build_graph(canonical_findings)
        
        # Print calibration analysis
        print("\n📊 CALIBRATION ANALYSIS")
        print("="*60)
        calibration_analysis = analyze_tool_calibration(canonical_findings)
        for tool, metrics in calibration_analysis.items():
            print(f"Tool: {tool}")
            print(f"  Samples: {metrics['sample_count']}")
            print(f"  Confidence: μ={metrics['confidence_mean']:.3f} σ={metrics['confidence_std']:.3f}")
            print(f"  Range: [{metrics['confidence_min']:.3f}, {metrics['confidence_max']:.3f}]")
            print(f"  Entropy: {metrics['entropy_mean']:.3f}")
        
        # Print graph analysis
        print("\n🔗 GRAPH ANALYSIS")
        print("="*60)
        print(f"Total nodes: {graph_analysis['total_nodes']}")
        print(f"Total edges (conflicts): {graph_analysis['total_edges']}")
        print(f"Conflict density: {graph_analysis['conflict_density']:.2%}")
        print(f"Critical conflicts: {graph_analysis['critical_conflicts']}")
        print(f"Moderate conflicts: {graph_analysis['moderate_conflicts']}")
        
        return graph_edges, graph_analysis
    
    def _save_tool_calls_with_conflicts(
        self, 
        tool_calls: List[ToolMessage], 
        canonical_findings: List[CanonicalFinding],
        conflicts: List,
        resolutions: List[Dict[str, Any]]
    ) -> None:
        """
        Save comprehensive tool execution log including conflict analysis.

        Args:
            tool_calls: Raw tool messages
            canonical_findings: Normalized findings
            conflicts: Detected conflicts
            resolutions: Conflict resolutions
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive log
        comprehensive_log = {
            "timestamp": datetime.now().isoformat(),
            "session_id": timestamp,
            "summary": {
                "total_tools_called": len(tool_calls),
                "canonical_findings": len(canonical_findings),
                "conflicts_detected": len(conflicts),
                "conflicts_resolved": len(resolutions),
                "human_review_required": any(r.get('should_defer', False) for r in resolutions)
            },
            "tool_executions": [],
            "canonical_findings": [],
            "conflict_analysis": {
                "conflicts": [],
                "resolutions": []
            }
        }
        
        # Add tool execution details
        for call in tool_calls:
            comprehensive_log["tool_executions"].append({
                "tool_call_id": call.tool_call_id,
                "tool_name": call.name,
                "args": call.args,
                "result": call.content[:500] + "..." if len(str(call.content)) > 500 else call.content,
                "timestamp": datetime.now().isoformat(),
            })
        
        # Add canonical findings
        for finding in canonical_findings:
            comprehensive_log["canonical_findings"].append(finding.to_dict())
        
        # Add conflict details
        for i, conflict in enumerate(conflicts):
            comprehensive_log["conflict_analysis"]["conflicts"].append(conflict.to_dict())
            if i < len(resolutions):
                comprehensive_log["conflict_analysis"]["resolutions"].append(resolutions[i])
        
        # Save to file
        filename = self.log_path / f"conflict_resolution_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(comprehensive_log, f, indent=2)
        
        print(f"📝 Comprehensive log saved: {filename}")

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
