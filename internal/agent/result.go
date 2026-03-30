package agent

import "github.com/fastclaw-ai/fastclaw/internal/provider"

// RunResult is the complete outcome of an agent run, providing full visibility
// into every step the agent took. Callers can use this for auditing, logging,
// analytics, or replaying the conversation.
//
// Mirrors the OpenAI Agents SDK's RunResult pattern.
type RunResult struct {
	// FinalOutput is the agent's final text response (empty if the run errored).
	FinalOutput string

	// Input is the original user input text.
	Input string

	// Messages is the complete message history including system, user,
	// assistant, and tool messages produced during the run.
	Messages []provider.Message

	// ToolCalls records every tool call made during the run, in order.
	ToolCalls []ToolCallRecord

	// TurnsUsed is the number of LLM invocations (turns) that occurred.
	TurnsUsed int

	// InputGuardrailResults captures the outcome of each input guardrail.
	InputGuardrailResults []GuardrailResult

	// OutputGuardrailResults captures the outcome of each output guardrail.
	OutputGuardrailResults []GuardrailResult

	// Error is non-nil if the run ended due to an error (max turns, guardrail
	// tripwire, LLM failure, etc.). FinalOutput may still contain a fallback
	// message in this case.
	Error error
}

// ToolCallRecord captures a single tool invocation for the RunResult.
type ToolCallRecord struct {
	Name      string
	Arguments string
	Result    string
	Error     error
}
