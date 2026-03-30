package agent

import (
	"fmt"

	"github.com/fastclaw-ai/fastclaw/internal/provider"
)

// AgentError is the base error type for all agent loop errors.
// It carries structured RunErrorDetails so callers can inspect the full
// context when something goes wrong.
//
// Mirrors the OpenAI Agents SDK's AgentsError pattern.
type AgentError struct {
	Err     error
	RunData *RunErrorDetails
}

func (e *AgentError) Error() string { return e.Err.Error() }
func (e *AgentError) Unwrap() error { return e.Err }

// RunErrorDetails captures the full context at the point of failure.
type RunErrorDetails struct {
	Input                string            // original user input
	Messages             []provider.Message // accumulated messages
	ToolResults          []toolResult       // tool results produced so far
	GuardrailResults     []GuardrailResult  // guardrail evaluations
	TurnsCompleted       int                // how many turns ran
}

// MaxTurnsExceededError indicates the agent loop exceeded maxToolIterations.
type MaxTurnsExceededError struct {
	*AgentError
	MaxTurns int
}

func NewMaxTurnsExceededError(maxTurns int, details *RunErrorDetails) *MaxTurnsExceededError {
	return &MaxTurnsExceededError{
		AgentError: &AgentError{
			Err:     fmt.Errorf("max turns (%d) exceeded", maxTurns),
			RunData: details,
		},
		MaxTurns: maxTurns,
	}
}

// ModelBehaviorError indicates the LLM did something unexpected, such as
// calling a non-existent tool or returning malformed JSON.
type ModelBehaviorError struct {
	*AgentError
}

func NewModelBehaviorError(msg string, details *RunErrorDetails) *ModelBehaviorError {
	return &ModelBehaviorError{
		AgentError: &AgentError{
			Err:     fmt.Errorf("model behavior error: %s", msg),
			RunData: details,
		},
	}
}
