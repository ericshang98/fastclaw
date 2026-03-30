package provider

import "context"

// Message represents a chat message.
type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content,omitempty"`
	ContentParts []ContentPart `json:"content_parts,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	Name         string        `json:"name,omitempty"`
}

// ContentPart represents a part of multimodal content.
type ContentPart struct {
	Type     string    `json:"type"`                // "text" or "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL for vision messages.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
}

// ToolCall represents a function call requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall contains the function name and arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool describes a tool available to the model.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a function tool.
type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

// Response is the result of a Chat call.
type Response struct {
	Content   string
	ToolCalls []ToolCall
}

// HasToolCalls returns true if the response contains tool calls.
func (r *Response) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// ToolEvent is emitted during the ReAct loop so SSE consumers can show
// real-time tool execution status (calling → completed/error).
type ToolEvent struct {
	Tool   string `json:"tool"`
	Status string `json:"status"` // "calling", "completed", "error"
	Params string `json:"params,omitempty"`
	Result string `json:"result,omitempty"`
}

// StreamChunk represents a single chunk from a streaming response.
type StreamChunk struct {
	Content   string
	ToolCalls []ToolCall
	ToolEvent *ToolEvent // non-nil = this chunk is a tool event, not text
	Done      bool
}

// StreamReader reads streaming chunks from an LLM response.
type StreamReader struct {
	ch  chan StreamChunk
	err error
}

// NewStreamReader creates a new StreamReader with the given channel.
func NewStreamReader(ch chan StreamChunk) *StreamReader {
	return &StreamReader{ch: ch}
}

// Next returns the next chunk and whether more chunks are available.
func (r *StreamReader) Next() (StreamChunk, bool) {
	chunk, ok := <-r.ch
	return chunk, ok
}

// Err returns any error that occurred during streaming.
func (r *StreamReader) Err() error {
	return r.err
}

// SetErr sets the error on the stream reader.
func (r *StreamReader) SetErr(err error) {
	r.err = err
}

// Provider is the LLM provider interface.
type Provider interface {
	Chat(ctx context.Context, messages []Message, tools []Tool, model string, maxTokens int, temperature float64) (*Response, error)
	ChatStream(ctx context.Context, messages []Message, tools []Tool, model string, maxTokens int, temperature float64) (*StreamReader, error)
}

// ToolChoiceProvider is an optional interface that providers can implement
// to support dynamic tool_choice control. The agent loop uses this to reset
// tool_choice after tools are used (OpenAI Agents SDK ResetToolChoice pattern).
type ToolChoiceProvider interface {
	// SetToolChoice sets tool_choice for subsequent Chat calls.
	// Valid values: "auto", "none", "required", or "" (provider default).
	SetToolChoice(choice string)
}
