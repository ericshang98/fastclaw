package agent

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
)

// GuardrailOutput is the result of a guardrail evaluation.
type GuardrailOutput struct {
	// Info contains guardrail-specific output data for debugging.
	Info any

	// TripwireTriggered halts execution immediately when true.
	TripwireTriggered bool
}

// InputGuardrail validates user input before the agent processes it.
// If TripwireTriggered is true, the agent loop returns immediately
// with an InputGuardrailTrippedError.
//
// Inspired by the OpenAI Agents SDK guardrail pattern.
type InputGuardrail struct {
	Name string
	Func func(ctx context.Context, agent *Agent, input string) (GuardrailOutput, error)
}

// OutputGuardrail validates the agent's final output before returning it.
type OutputGuardrail struct {
	Name string
	Func func(ctx context.Context, agent *Agent, output string) (GuardrailOutput, error)
}

// GuardrailResult captures the outcome of a guardrail evaluation.
type GuardrailResult struct {
	Name    string
	Output  GuardrailOutput
	Error   error
}

// runInputGuardrails executes all input guardrails in parallel.
// Returns the first tripwire-triggered result, or nil if all pass.
func runInputGuardrails(ctx context.Context, agent *Agent, input string, guardrails []InputGuardrail) (*GuardrailResult, error) {
	if len(guardrails) == 0 {
		return nil, nil
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]GuardrailResult, len(guardrails))
	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	for i, g := range guardrails {
		go func(idx int, gr InputGuardrail) {
			defer wg.Done()
			out, err := gr.Func(ctx, agent, input)
			results[idx] = GuardrailResult{Name: gr.Name, Output: out, Error: err}
			if err != nil || out.TripwireTriggered {
				cancel() // stop other guardrails
			}
		}(i, g)
	}
	wg.Wait()

	// Check results
	for _, r := range results {
		if r.Error != nil {
			slog.Warn("input guardrail error", "name", r.Name, "error", r.Error)
			return &r, r.Error
		}
		if r.Output.TripwireTriggered {
			slog.Warn("input guardrail tripwire triggered", "name", r.Name)
			return &r, nil
		}
	}
	return nil, nil
}

// runOutputGuardrails executes all output guardrails in parallel.
func runOutputGuardrails(ctx context.Context, agent *Agent, output string, guardrails []OutputGuardrail) (*GuardrailResult, error) {
	if len(guardrails) == 0 {
		return nil, nil
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]GuardrailResult, len(guardrails))
	var wg sync.WaitGroup
	wg.Add(len(guardrails))

	for i, g := range guardrails {
		go func(idx int, gr OutputGuardrail) {
			defer wg.Done()
			out, err := gr.Func(ctx, agent, output)
			results[idx] = GuardrailResult{Name: gr.Name, Output: out, Error: err}
			if err != nil || out.TripwireTriggered {
				cancel()
			}
		}(i, g)
	}
	wg.Wait()

	for _, r := range results {
		if r.Error != nil {
			slog.Warn("output guardrail error", "name", r.Name, "error", r.Error)
			return &r, r.Error
		}
		if r.Output.TripwireTriggered {
			slog.Warn("output guardrail tripwire triggered", "name", r.Name)
			return &r, nil
		}
	}
	return nil, nil
}

// InputGuardrailTrippedError indicates an input guardrail rejected the input.
type InputGuardrailTrippedError struct {
	GuardrailName string
	Output        GuardrailOutput
}

func (e *InputGuardrailTrippedError) Error() string {
	return fmt.Sprintf("input guardrail %q tripwire triggered", e.GuardrailName)
}

// OutputGuardrailTrippedError indicates an output guardrail rejected the output.
type OutputGuardrailTrippedError struct {
	GuardrailName string
	Output        GuardrailOutput
}

func (e *OutputGuardrailTrippedError) Error() string {
	return fmt.Sprintf("output guardrail %q tripwire triggered", e.GuardrailName)
}
