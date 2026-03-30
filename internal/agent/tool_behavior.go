package agent

import "context"

// ToolUseBehavior controls what happens after tool calls are executed.
// By default (RunLLMAgain), results are fed back to the LLM for further
// reasoning. Alternative behaviors let you short-circuit the loop and
// use tool output directly as the final response.
//
// Mirrors the OpenAI Agents SDK's ToolUseBehavior pattern.
type ToolUseBehavior interface {
	// ToolsToFinalOutput examines tool results and decides whether to
	// treat them as the final output or continue the loop.
	ToolsToFinalOutput(ctx context.Context, results []toolResult) (ToolBehaviorResult, error)
}

// ToolBehaviorResult is the decision from a ToolUseBehavior.
type ToolBehaviorResult struct {
	// IsFinalOutput is true when the tool output should be returned
	// directly without another LLM call.
	IsFinalOutput bool

	// FinalOutput is the content to return. Only meaningful when
	// IsFinalOutput is true.
	FinalOutput string
}

var notFinalOutput = ToolBehaviorResult{IsFinalOutput: false}

// ── Built-in behaviors ──────────────────────────────────────────────

// RunLLMAgainBehavior is the default: always feed tool results back to the LLM.
func RunLLMAgainBehavior() ToolUseBehavior { return runLLMAgain{} }

type runLLMAgain struct{}

func (runLLMAgain) ToolsToFinalOutput(context.Context, []toolResult) (ToolBehaviorResult, error) {
	return notFinalOutput, nil
}

// StopOnFirstToolBehavior uses the first tool's output as the final response,
// skipping the LLM summarization step. Useful for tools like get_weather
// where the raw result is already user-facing.
func StopOnFirstToolBehavior() ToolUseBehavior { return stopOnFirstTool{} }

type stopOnFirstTool struct{}

func (stopOnFirstTool) ToolsToFinalOutput(_ context.Context, results []toolResult) (ToolBehaviorResult, error) {
	if len(results) == 0 {
		return notFinalOutput, nil
	}
	return ToolBehaviorResult{IsFinalOutput: true, FinalOutput: results[0].Content}, nil
}

// StopAtToolsBehavior stops the loop if any of the named tools are called,
// using that tool's output as the final response.
func StopAtToolsBehavior(toolNames ...string) ToolUseBehavior {
	nameSet := make(map[string]bool, len(toolNames))
	for _, n := range toolNames {
		nameSet[n] = true
	}
	return stopAtTools{names: nameSet}
}

type stopAtTools struct {
	names map[string]bool
}

func (s stopAtTools) ToolsToFinalOutput(_ context.Context, results []toolResult) (ToolBehaviorResult, error) {
	for _, r := range results {
		if s.names[r.Name] {
			return ToolBehaviorResult{IsFinalOutput: true, FinalOutput: r.Content}, nil
		}
	}
	return notFinalOutput, nil
}

// ToolsToFinalOutputFunc is a custom function-based ToolUseBehavior.
// Use this when the built-in behaviors don't fit your needs.
type ToolsToFinalOutputFunc func(ctx context.Context, results []toolResult) (ToolBehaviorResult, error)

func (f ToolsToFinalOutputFunc) ToolsToFinalOutput(ctx context.Context, results []toolResult) (ToolBehaviorResult, error) {
	return f(ctx, results)
}
