package agent

import (
	"context"
	"crypto/sha256"
	"log/slog"
	"sync"

	"github.com/fastclaw-ai/fastclaw/internal/bus"
	"github.com/fastclaw-ai/fastclaw/internal/provider"
	"github.com/fastclaw-ai/fastclaw/internal/session"
)

// NextStep represents the decision after processing one LLM turn.
// Inspired by OpenAI Agents SDK's state machine pattern.
type NextStep interface{ nextStep() }

// NextStepFinalOutput indicates the agent produced a final text response.
type NextStepFinalOutput struct {
	Content string
}

// NextStepRunAgain indicates tool calls were executed; the LLM should be
// invoked again with the updated message history.
type NextStepRunAgain struct{}

// NextStepLoopDetected indicates a tool loop was detected and the agent
// should stop iterating.
type NextStepLoopDetected struct{}

func (NextStepFinalOutput) nextStep()  {}
func (NextStepRunAgain) nextStep()     {}
func (NextStepLoopDetected) nextStep() {}

// stepResult bundles the next-step decision with the updated message history.
type stepResult struct {
	NextStep NextStep
	Messages []provider.Message
}

// RunConfig holds per-run configuration for the agent loop.
type RunConfig struct {
	InputGuardrails  []InputGuardrail
	OutputGuardrails []OutputGuardrail

	// ResetToolChoice resets tool_choice to nil after tools are used,
	// preventing infinite tool-calling loops. Default: true.
	// Mirrors OpenAI Agents SDK's ResetToolChoice behavior.
	ResetToolChoice *bool
}

// shouldResetToolChoice returns whether tool_choice should be reset after use.
func (rc *RunConfig) shouldResetToolChoice() bool {
	if rc == nil || rc.ResetToolChoice == nil {
		return true // default: reset
	}
	return *rc.ResetToolChoice
}

// toolUseTracker tracks whether tools have been used in the current run,
// used to decide when to reset tool_choice (OpenAI Agents SDK pattern).
type toolUseTracker struct {
	used bool
}

func (t *toolUseTracker) markUsed()    { t.used = true }
func (t *toolUseTracker) hasUsedTools() bool { return t.used }

// loopDetector tracks consecutive identical tool calls to break infinite loops.
type loopDetector struct {
	lastName string
	lastHash [32]byte
	count    int
}

func (ld *loopDetector) check(name, args string) bool {
	hash := sha256.Sum256([]byte(args))
	if name == ld.lastName && hash == ld.lastHash {
		ld.count++
	} else {
		ld.count = 1
		ld.lastName = name
		ld.lastHash = hash
	}
	return ld.count >= 3
}

// toolResult holds the output of a single tool execution.
type toolResult struct {
	ToolCallID string
	Name       string
	Content    string
	Error      error
}

// statefulTools are tools that mutate state and must execute serially.
// All other tools are considered stateless and may execute in parallel.
var statefulTools = map[string]bool{
	"exec":            true,
	"write_file":      true,
	"message":         true,
	"generate_image":  true,
	"spawn_subagent":  true,
	"create_cron_job": true,
	"delete_cron_job": true,
}

// processToolCalls classifies, executes, and appends tool results.
// Stateless tools (read_file, list_dir, web_fetch, web_search, memory_search,
// load_skill, list_cron_jobs) run in parallel via goroutines.
// Stateful tools (exec, write_file, message, …) run serially to avoid
// race conditions on mutable state.
//
// sendEvent is optional — pass nil for non-streaming callers.
func (a *Agent) processToolCalls(
	ctx context.Context,
	toolCalls []provider.ToolCall,
	messages []provider.Message,
	sess *session.Session,
	msg bus.InboundMessage,
	ld *loopDetector,
	sendEvent func(provider.ToolEvent),
) ([]provider.Message, NextStep) {

	// ── Classify ──
	type classified struct {
		tc       provider.ToolCall
		stateful bool
	}
	var plan []classified
	for _, tc := range toolCalls {
		// Loop detection — check before execution
		if ld.check(tc.Function.Name, tc.Function.Arguments) {
			slog.Warn("tool loop detected", "agent", a.name, "tool", tc.Function.Name)
			warnMsg := provider.Message{
				Role:    "system",
				Content: "Loop detected: you called the same tool with the same arguments 3 times. Please try a different approach.",
			}
			sess.Append(warnMsg)
			messages = append(messages, warnMsg)
			return messages, NextStepLoopDetected{}
		}
		plan = append(plan, classified{tc: tc, stateful: statefulTools[tc.Function.Name]})
	}

	// ── Execute stateless tools in parallel ──
	var stateless, stateful []classified
	for _, c := range plan {
		if c.stateful {
			stateful = append(stateful, c)
		} else {
			stateless = append(stateless, c)
		}
	}

	var results []toolResult

	if len(stateless) > 0 {
		parallel := make([]toolResult, len(stateless))
		var wg sync.WaitGroup
		wg.Add(len(stateless))
		for i, c := range stateless {
			go func(idx int, tc provider.ToolCall) {
				defer wg.Done()
				parallel[idx] = a.executeTool(ctx, tc, sendEvent)
			}(i, c.tc)
		}
		wg.Wait()
		results = append(results, parallel...)
	}

	// ── Execute stateful tools serially ──
	for _, c := range stateful {
		results = append(results, a.executeTool(ctx, c.tc, sendEvent))
	}

	// ── Append all results to messages & handle MEDIA: protocol ──
	for _, r := range results {
		toolMsg := provider.Message{
			Role:       "tool",
			Content:    r.Content,
			ToolCallID: r.ToolCallID,
			Name:       r.Name,
		}
		sess.Append(toolMsg)
		messages = append(messages, toolMsg)

		if mediaPaths := extractMediaPaths(r.Content); len(mediaPaths) > 0 {
			a.sendMediaFiles(msg, mediaPaths)
		}
	}

	return messages, NextStepRunAgain{}
}

// executeTool runs a single tool call with hooks and event emission.
func (a *Agent) executeTool(
	ctx context.Context,
	tc provider.ToolCall,
	sendEvent func(provider.ToolEvent),
) toolResult {
	// Emit "calling" event
	if sendEvent != nil {
		sendEvent(provider.ToolEvent{
			Tool:   tc.Function.Name,
			Status: "calling",
			Params: tc.Function.Arguments,
		})
	}

	// Hook: BeforeToolCall
	hcBefore := &HookContext{
		AgentName: a.name,
		Point:     BeforeToolCall,
		ToolName:  tc.Function.Name,
		ToolArgs:  tc.Function.Arguments,
	}
	a.hooks.Run(ctx, hcBefore)

	slog.Info("executing tool", "agent", a.name, "name", tc.Function.Name, "id", tc.ID)

	result, execErr := a.registry.Execute(ctx, tc.Function.Name, tc.Function.Arguments)

	// Hook: AfterToolCall
	hcAfter := &HookContext{
		AgentName:  a.name,
		Point:      AfterToolCall,
		ToolName:   tc.Function.Name,
		ToolResult: result,
		Error:      execErr,
		StartTime:  hcBefore.StartTime,
	}
	a.hooks.Run(ctx, hcAfter)

	if execErr != nil {
		slog.Warn("tool execution error", "agent", a.name, "name", tc.Function.Name, "error", execErr)
	}

	// Emit "completed" or "error" event
	if sendEvent != nil {
		status := "completed"
		eventResult := result
		if execErr != nil {
			status = "error"
			eventResult = execErr.Error()
		}
		if len(eventResult) > 2000 {
			eventResult = eventResult[:2000] + "…"
		}
		sendEvent(provider.ToolEvent{
			Tool:   tc.Function.Name,
			Status: status,
			Params: tc.Function.Arguments,
			Result: eventResult,
		})
	}

	return toolResult{
		ToolCallID: tc.ID,
		Name:       tc.Function.Name,
		Content:    result,
		Error:      execErr,
	}
}
