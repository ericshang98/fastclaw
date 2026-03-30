package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/fastclaw-ai/fastclaw/internal/agent/tools"
	"github.com/fastclaw-ai/fastclaw/internal/bus"
	"github.com/fastclaw-ai/fastclaw/internal/config"
	"github.com/fastclaw-ai/fastclaw/internal/mcp"
	"github.com/fastclaw-ai/fastclaw/internal/provider"
	"github.com/fastclaw-ai/fastclaw/internal/session"
)

// Agent is the agentic loop, using a state-machine pattern inspired by the
// OpenAI Agents SDK. Each iteration returns a NextStep that drives the loop.
type Agent struct {
	name              string
	provider          provider.Provider
	registry          *tools.Registry
	sessions          *session.Manager
	memory            *Memory
	ctxBuilder        *ContextBuilder
	mcpMgr            *mcp.Manager
	hooks             *HookRegistry
	model             string
	maxTokens         int
	temperature       float64
	maxToolIterations int
	thinking          string
	workspacePath     string
	homeDir           string
	skillsCfg         config.SkillsConfig
	globalSkillsCfg   config.SkillsCfg
	messageBus        *bus.MessageBus
	subAgentSpawner   tools.SubAgentSpawner

	// runConfig holds optional guardrails and loop behavior settings.
	runConfig *RunConfig
}

// SetRunConfig sets optional per-agent run configuration (guardrails, etc.).
func (a *Agent) SetRunConfig(rc *RunConfig) {
	a.runConfig = rc
}

// NewAgent creates a new Agent from a resolved config.
func NewAgent(rc config.ResolvedAgent, prov provider.Provider, mb *bus.MessageBus, homeDir string) *Agent {
	return NewAgentWithSkillsCfg(rc, prov, mb, homeDir, config.SkillsCfg{})
}

// NewAgentWithSkillsCfg creates a new Agent with global skills config for env injection.
func NewAgentWithSkillsCfg(rc config.ResolvedAgent, prov provider.Provider, mb *bus.MessageBus, homeDir string, globalSkillsCfg config.SkillsCfg) *Agent {
	memory := NewMemory(rc.Workspace)
	registry := tools.NewRegistry(rc.Workspace)
	tools.RegisterMessage(registry, mb)
	tools.RegisterMemorySearch(registry, rc.Workspace)
	tools.RegisterWebFetch(registry)
	tools.RegisterLoadSkill(registry, homeDir, rc.Workspace, "")

	// Register image generation tool if endpoint is configured
	if imgURL := os.Getenv("IMAGE_GEN_URL"); imgURL != "" {
		imgAuth := os.Getenv("IMAGE_GEN_AUTH")
		tools.RegisterGenerateImage(registry, imgURL, imgAuth, rc.ID)
		slog.Info("registered generate_image tool", "agent", rc.ID)
	}

	// Load skills with OpenClaw compatibility
	loader := NewSkillsLoaderWithGlobal(homeDir, rc.Workspace, "", rc.Skills, globalSkillsCfg)
	skills := loader.LoadSkills()
	skillsSummary := loader.BuildSkillsSummary(skills)

	// Set up skill env injection for exec tool
	skillDirs := loader.AllSkillDirs()
	tools.RegisterExecWithSkillEnv(registry, nil, loader.SkillEnvVars, skillDirs)

	if len(skills) > 0 {
		slog.Info("loaded skills", "agent", rc.ID, "count", len(skills))
	}

	// Set up hooks with logging
	hooks := NewHookRegistry()
	hooks.Register(BeforeModelCall, LoggingHook())
	hooks.Register(AfterModelCall, LoggingHook())
	hooks.Register(BeforeToolCall, LoggingHook())
	hooks.Register(AfterToolCall, LoggingHook())
	hooks.Register(OnAgentStart, LoggingHook())
	hooks.Register(OnAgentEnd, LoggingHook())

	ag := &Agent{
		name:              rc.ID,
		provider:          prov,
		registry:          registry,
		sessions:          session.NewManager(rc.Workspace + "/sessions"),
		memory:            memory,
		ctxBuilder:        newContextBuilderWithThinking(rc.Workspace, memory, skillsSummary, rc.Thinking),
		hooks:             hooks,
		model:             rc.Model,
		maxTokens:         rc.MaxTokens,
		temperature:       rc.Temperature,
		maxToolIterations: rc.MaxToolIterations,
		thinking:          rc.Thinking,
		workspacePath:     rc.Workspace,
		homeDir:           homeDir,
		skillsCfg:         rc.Skills,
		globalSkillsCfg:   globalSkillsCfg,
		messageBus:        mb,
	}

	// Connect MCP servers and register their tools
	if len(rc.MCPServers) > 0 {
		mcpMgr := mcp.NewManager(rc.MCPServers)
		ag.mcpMgr = mcpMgr

		for _, td := range mcpMgr.ToolDefs() {
			toolName := td.Name
			ag.registry.Register(toolName, td.Description, td.InputSchema,
				func(ctx context.Context, args json.RawMessage) (string, error) {
					return mcpMgr.CallTool(ctx, toolName, args)
				},
			)
		}

		if mcpMgr.HasTools() {
			slog.Info("registered MCP tools", "agent", rc.ID)
		}
	}

	return ag
}

func newContextBuilderWithThinking(workspace string, memory *Memory, skillsSummary string, thinking string) *ContextBuilder {
	cb := NewContextBuilder(workspace, memory, skillsSummary)
	if thinking != "" {
		cb.SetThinking(thinking)
	}
	return cb
}

// Name returns the agent's name.
func (a *Agent) Name() string {
	return a.name
}

// HandleWebChat handles a chat message from the web UI.
func (a *Agent) HandleWebChat(ctx context.Context, text string) string {
	msg := bus.InboundMessage{
		Channel:  "web",
		ChatID:   "web-ui",
		UserID:   "web-user",
		Text:     text,
		PeerKind: "dm",
	}
	return a.HandleMessage(ctx, msg)
}

// workspace returns the agent's workspace path.
func (a *Agent) workspace() string {
	return a.workspacePath
}

// SetGroupContext configures group chat awareness for this agent's system prompt.
func (a *Agent) SetGroupContext(gc *GroupContext) {
	a.ctxBuilder.SetGroupContext(gc)
}

// InjectGroupMessage appends a message from another bot into the session history
// without triggering an LLM call. This gives the agent awareness of what other
// bots said in the group chat.
func (a *Agent) InjectGroupMessage(ctx context.Context, msg bus.InboundMessage) {
	sess := a.sessions.Get(msg.Channel, msg.ChatID)
	label := msg.SenderName
	if label == "" {
		label = "Bot"
	}
	content := fmt.Sprintf("[%s]: %s", label, msg.Text)
	sess.Append(provider.Message{Role: "user", Content: content})
}

// SetSubAgentSpawner sets the sub-agent spawner for the spawn_subagent tool.
func (a *Agent) SetSubAgentSpawner(spawner tools.SubAgentSpawner) {
	a.subAgentSpawner = spawner
	tools.RegisterSubAgent(a.registry, spawner, a.name)
}

// ToolRegistry returns the agent's tool registry for external registration.
func (a *Agent) ToolRegistry() *tools.Registry {
	return a.registry
}

// RegisterWebSearchTool registers the web_search tool with the given API key.
func (a *Agent) RegisterWebSearchTool(apiKey string) {
	tools.RegisterWebSearch(a.registry, apiKey)
}

// Sessions returns the session manager for this agent.
func (a *Agent) Sessions() *session.Manager {
	return a.sessions
}

// Model returns the agent's model name.
func (a *Agent) Model() string {
	return a.model
}

// HandleMessage processes an inbound message and returns the final text.
// It is a convenience wrapper around HandleMessageFull.
func (a *Agent) HandleMessage(ctx context.Context, msg bus.InboundMessage) string {
	if result := a.handleSlashCommand(msg); result.handled {
		return result.reply
	}
	r := a.HandleMessageFull(ctx, msg)
	return r.FinalOutput
}

// HandleMessageFull processes an inbound message through the agent loop and
// returns a full RunResult with all intermediate data.
//
// The loop uses a state-machine pattern inspired by the OpenAI Agents SDK:
// each iteration returns a NextStep that determines whether to return a final
// response, execute more tools, or stop due to a detected loop.
func (a *Agent) HandleMessageFull(ctx context.Context, msg bus.InboundMessage) *RunResult {
	rr := &RunResult{Input: msg.Text}

	sess, messages, toolDefs := a.prepareContext(msg)

	// ── Hook: OnAgentStart ──
	a.hooks.Run(ctx, &HookContext{AgentName: a.name, Point: OnAgentStart})

	// ── Input guardrails (parallel) ──
	if a.runConfig != nil && len(a.runConfig.InputGuardrails) > 0 {
		result, err := runInputGuardrails(ctx, a, msg.Text, a.runConfig.InputGuardrails)
		if result != nil {
			rr.InputGuardrailResults = append(rr.InputGuardrailResults, *result)
		}
		if err != nil {
			slog.Error("input guardrail error", "agent", a.name, "error", err)
			rr.FinalOutput = "Sorry, I can't process that request."
			rr.Error = &InputGuardrailTrippedError{GuardrailName: result.Name, Output: result.Output}
			rr.Messages = messages
			return rr
		}
		if result != nil && result.Output.TripwireTriggered {
			slog.Warn("input guardrail tripped", "agent", a.name, "guardrail", result.Name)
			rr.FinalOutput = "Sorry, I can't process that request."
			rr.Error = &InputGuardrailTrippedError{GuardrailName: result.Name, Output: result.Output}
			rr.Messages = messages
			return rr
		}
	}

	ld := &loopDetector{}
	tracker := &toolUseTracker{}

	for turn := 1; turn <= a.maxToolIterations; turn++ {
		rr.TurnsUsed = turn

		slog.Info("agent loop iteration",
			"agent", a.name, "turn", turn,
			"channel", msg.Channel, "chat_id", msg.ChatID,
		)

		// ── ResetToolChoice ──
		a.maybeResetToolChoice(tracker)

		// ── Call LLM ──
		hcBefore := &HookContext{AgentName: a.name, Point: BeforeModelCall, Messages: messages}
		a.hooks.Run(ctx, hcBefore)

		resp, err := a.provider.Chat(ctx, messages, toolDefs, a.model, a.maxTokens, a.temperature)

		hcAfter := &HookContext{AgentName: a.name, Point: AfterModelCall, Messages: messages, Response: resp, Error: err, StartTime: hcBefore.StartTime}
		a.hooks.Run(ctx, hcAfter)

		if err != nil {
			slog.Error("LLM chat failed", "agent", a.name, "error", err)
			rr.FinalOutput = "Sorry, I encountered an error processing your request."
			rr.Error = err
			rr.Messages = messages
			return rr
		}

		// ── Decide next step ──
		step := a.processLLMResponse(ctx, resp, messages, sess, msg, ld, nil)

		switch s := step.NextStep.(type) {
		case NextStepFinalOutput:
			// ── Output guardrails (parallel) ──
			if a.runConfig != nil && len(a.runConfig.OutputGuardrails) > 0 {
				result, err := runOutputGuardrails(ctx, a, s.Content, a.runConfig.OutputGuardrails)
				if result != nil {
					rr.OutputGuardrailResults = append(rr.OutputGuardrailResults, *result)
				}
				if err != nil {
					slog.Error("output guardrail error", "agent", a.name, "error", err)
					rr.FinalOutput = "Sorry, I encountered an error validating the response."
					rr.Error = err
					rr.Messages = step.Messages
					return rr
				}
				if result != nil && result.Output.TripwireTriggered {
					slog.Warn("output guardrail tripped", "agent", a.name, "guardrail", result.Name)
					rr.FinalOutput = "Sorry, I can't provide that response."
					rr.Error = &OutputGuardrailTrippedError{GuardrailName: result.Name, Output: result.Output}
					rr.Messages = step.Messages
					return rr
				}
			}

			rr.FinalOutput = s.Content
			rr.Messages = step.Messages
			// ── Hook: OnAgentEnd ──
			a.hooks.Run(ctx, &HookContext{AgentName: a.name, Point: OnAgentEnd, FinalOutput: s.Content})
			return rr

		case NextStepRunAgain:
			tracker.markUsed()
			messages = step.Messages
			continue

		case NextStepLoopDetected:
			slog.Warn("breaking agent loop due to loop detection", "agent", a.name)
		}
		break
	}

	// ── Max turns exceeded ──
	slog.Warn("max tool iterations reached", "agent", a.name, "max", a.maxToolIterations)

	// Allow custom handling via OnMaxTurns callback
	fallback := "I've reached the maximum number of tool iterations. Here's what I have so far."
	if a.runConfig != nil && a.runConfig.OnMaxTurns != nil {
		if custom := a.runConfig.OnMaxTurns(ctx, messages); custom != "" {
			fallback = custom
		}
	}

	rr.FinalOutput = fallback
	rr.Error = NewMaxTurnsExceededError(a.maxToolIterations, &RunErrorDetails{
		Input:          msg.Text,
		Messages:       messages,
		TurnsCompleted: rr.TurnsUsed,
	})
	rr.Messages = messages
	a.hooks.Run(ctx, &HookContext{AgentName: a.name, Point: OnAgentEnd, FinalOutput: fallback})
	return rr
}

// maybeResetToolChoice resets tool_choice to "auto" after tools have been used,
// preventing infinite tool-calling loops. Mirrors the OpenAI Agents SDK's
// MaybeResetToolChoice behavior.
func (a *Agent) maybeResetToolChoice(tracker *toolUseTracker) {
	if !tracker.hasUsedTools() {
		return
	}
	if a.runConfig != nil && !a.runConfig.shouldResetToolChoice() {
		return
	}
	if tcp, ok := a.provider.(provider.ToolChoiceProvider); ok {
		tcp.SetToolChoice("auto")
	}
}

// prepareContext builds the session, messages array, and tool definitions
// shared by both HandleMessage and HandleMessageStream.
func (a *Agent) prepareContext(msg bus.InboundMessage) (*session.Session, []provider.Message, []provider.Tool) {
	sess := a.sessions.Get(msg.Channel, msg.ChatID)

	a.hooks.Run(context.Background(), &HookContext{AgentName: a.name, Point: BeforeSystemPrompt})
	systemPrompt := a.ctxBuilder.BuildSystemPrompt()
	a.hooks.Run(context.Background(), &HookContext{AgentName: a.name, Point: AfterSystemPrompt})

	runtimeCtx := a.ctxBuilder.BuildRuntimeContext(msg.Channel, msg.ChatID)
	userContent := runtimeCtx + "\n\n" + msg.Text

	userMsg := provider.Message{Role: "user", Content: userContent}
	if msg.PhotoURL != "" {
		userMsg.Content = ""
		userMsg.ContentParts = []provider.ContentPart{
			{Type: "text", Text: userContent},
			{Type: "image_url", ImageURL: &provider.ImageURL{URL: msg.PhotoURL, Detail: "auto"}},
		}
	}
	sess.Append(userMsg)

	// Context compaction
	sessionMsgs := sess.GetMessages()
	compactResult, err := CompactMessages(sessionMsgs, a.workspacePath, a.provider, a.model)
	if err != nil {
		slog.Warn("compaction error", "agent", a.name, "error", err)
	}
	if compactResult != nil && compactResult.Pruned {
		sess.ReplaceMessages(compactResult.Messages)
		sessionMsgs = compactResult.Messages
		slog.Info("context compacted", "agent", a.name, "log_file", compactResult.LogFile)
	}

	messages := make([]provider.Message, 0, len(sessionMsgs)+1)
	messages = append(messages, provider.Message{Role: "system", Content: systemPrompt})
	messages = append(messages, sessionMsgs...)

	return sess, messages, a.registry.Definitions()
}

// processLLMResponse examines the LLM response and returns a stepResult:
//   - No tool calls → NextStepFinalOutput
//   - Tool calls present → execute them (parallel/serial), return NextStepRunAgain
//   - Loop detected → NextStepLoopDetected
func (a *Agent) processLLMResponse(
	ctx context.Context,
	resp *provider.Response,
	messages []provider.Message,
	sess *session.Session,
	msg bus.InboundMessage,
	ld *loopDetector,
	sendEvent func(provider.ToolEvent),
) stepResult {
	// No tool calls → final output
	if !resp.HasToolCalls() {
		sess.Append(provider.Message{Role: "assistant", Content: resp.Content})
		return stepResult{
			NextStep: NextStepFinalOutput{Content: resp.Content},
			Messages: messages,
		}
	}

	// Append assistant message with tool calls
	assistantMsg := provider.Message{
		Role:      "assistant",
		Content:   resp.Content,
		ToolCalls: resp.ToolCalls,
	}
	sess.Append(assistantMsg)
	messages = append(messages, assistantMsg)

	// Execute tools — stateless in parallel, stateful serially
	messages, next := a.processToolCalls(ctx, resp.ToolCalls, messages, sess, msg, ld, sendEvent)

	return stepResult{NextStep: next, Messages: messages}
}

// HandleMessageStream processes a message through the agent loop and returns
// a StreamReader for the final response. Tool call iterations use non-streaming
// Chat; tool_event chunks are emitted so SSE consumers can show real-time
// progress. The final text response uses ChatStream for true SSE streaming.
func (a *Agent) HandleMessageStream(ctx context.Context, msg bus.InboundMessage) *provider.StreamReader {
	if result := a.handleSlashCommand(msg); result.handled {
		return a.stringStream(result.reply)
	}

	sess, messages, toolDefs := a.prepareContext(msg)

	outCh := make(chan provider.StreamChunk, 64)
	outReader := provider.NewStreamReader(outCh)

	sendEvent := func(te provider.ToolEvent) {
		select {
		case outCh <- provider.StreamChunk{ToolEvent: &te}:
		case <-ctx.Done():
		}
	}

	go func() {
		defer close(outCh)

		// ── Input guardrails (parallel) ──
		if a.runConfig != nil && len(a.runConfig.InputGuardrails) > 0 {
			result, err := runInputGuardrails(ctx, a, msg.Text, a.runConfig.InputGuardrails)
			if err != nil || (result != nil && result.Output.TripwireTriggered) {
				outCh <- provider.StreamChunk{Content: "Sorry, I can't process that request.", Done: true}
				return
			}
		}

		ld := &loopDetector{}
		tracker := &toolUseTracker{}

		for turn := 1; turn <= a.maxToolIterations; turn++ {
			// ── ResetToolChoice ──
			a.maybeResetToolChoice(tracker)

			// ── Call LLM (non-streaming) to check for tool calls ──
			hcBefore := &HookContext{AgentName: a.name, Point: BeforeModelCall, Messages: messages}
			a.hooks.Run(ctx, hcBefore)

			resp, err := a.provider.Chat(ctx, messages, toolDefs, a.model, a.maxTokens, a.temperature)

			hcAfter := &HookContext{AgentName: a.name, Point: AfterModelCall, Messages: messages, Response: resp, Error: err, StartTime: hcBefore.StartTime}
			a.hooks.Run(ctx, hcAfter)

			if err != nil {
				slog.Error("LLM chat failed", "agent", a.name, "error", err)
				outCh <- provider.StreamChunk{Content: "Sorry, I encountered an error processing your request.", Done: true}
				return
			}

			// ── No tool calls → stream the final text response ──
			if !resp.HasToolCalls() {
				// ── Output guardrails ──
				if a.runConfig != nil && len(a.runConfig.OutputGuardrails) > 0 {
					result, gErr := runOutputGuardrails(ctx, a, resp.Content, a.runConfig.OutputGuardrails)
					if gErr != nil || (result != nil && result.Output.TripwireTriggered) {
						outCh <- provider.StreamChunk{Content: "Sorry, I can't provide that response.", Done: true}
						return
					}
				}

				sr, err := a.provider.ChatStream(ctx, messages, toolDefs, a.model, a.maxTokens, a.temperature)
				if err != nil {
					slog.Error("LLM stream failed, falling back", "agent", a.name, "error", err)
					sess.Append(provider.Message{Role: "assistant", Content: resp.Content})
					outCh <- provider.StreamChunk{Content: resp.Content, Done: true}
					return
				}
				var full strings.Builder
				for {
					chunk, ok := sr.Next()
					if !ok {
						break
					}
					if chunk.Content != "" {
						full.WriteString(chunk.Content)
					}
					select {
					case outCh <- chunk:
					case <-ctx.Done():
						return
					}
				}
				sess.Append(provider.Message{Role: "assistant", Content: full.String()})
				return
			}

			// ── Tool calls → process via state machine ──
			step := a.processLLMResponse(ctx, resp, messages, sess, msg, ld, sendEvent)

			switch step.NextStep.(type) {
			case NextStepRunAgain:
				tracker.markUsed()
				messages = step.Messages
				continue
			default:
				break
			}
			break
		}

		outCh <- provider.StreamChunk{Content: "I've reached the maximum number of tool iterations. Here's what I have so far.", Done: true}
	}()

	return outReader
}

// stringStream creates a StreamReader that yields a single string.
func (a *Agent) stringStream(text string) *provider.StreamReader {
	ch := make(chan provider.StreamChunk, 2)
	go func() {
		ch <- provider.StreamChunk{Content: text, Done: true}
		close(ch)
	}()
	return provider.NewStreamReader(ch)
}

// WorkspacePath returns the agent's workspace directory.
func (a *Agent) WorkspacePath() string {
	return a.workspacePath
}

// UpdateConfig updates the agent's runtime config (model, temperature, etc.)
func (a *Agent) UpdateConfig(rc config.ResolvedAgent) {
	a.model = rc.Model
	a.maxTokens = rc.MaxTokens
	a.temperature = rc.Temperature
	a.maxToolIterations = rc.MaxToolIterations
}

// ReloadWorkspaceFiles re-reads workspace .md files (SOUL.md, AGENTS.md, etc.)
// and rebuilds the context builder.
func (a *Agent) ReloadWorkspaceFiles() {
	a.memory = NewMemory(a.workspacePath)
	// Rebuild skills summary
	loader := NewSkillsLoaderWithGlobal(a.homeDir, a.workspacePath, "", a.skillsCfg, a.globalSkillsCfg)
	skills := loader.LoadSkills()
	skillsSummary := loader.BuildSkillsSummary(skills)
	a.ctxBuilder = NewContextBuilder(a.workspacePath, a.memory, skillsSummary)
}

// extractMediaPaths scans tool output for MEDIA: lines and returns file paths.
// The MEDIA: protocol is used by OpenClaw skills to attach files to chat messages.
func extractMediaPaths(output string) []string {
	var paths []string
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "MEDIA:") {
			path := strings.TrimSpace(strings.TrimPrefix(line, "MEDIA:"))
			if path != "" {
				if _, err := os.Stat(path); err == nil {
					paths = append(paths, path)
				}
			}
		}
	}
	return paths
}

// sendMediaFiles sends extracted MEDIA: files to the outbound bus.
func (a *Agent) sendMediaFiles(msg bus.InboundMessage, mediaPaths []string) {
	if len(mediaPaths) == 0 || a.messageBus == nil {
		return
	}
	outMsg := bus.OutboundMessage{
		Channel:    msg.Channel,
		AccountID:  msg.AccountID,
		ChatID:     msg.ChatID,
		MediaPaths: mediaPaths,
	}
	select {
	case a.messageBus.Outbound <- outMsg:
	default:
		slog.Warn("outbound channel full, dropping media message", "agent", a.name)
	}
}
