package agent

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fastclaw-ai/fastclaw/internal/bus"
)

const (
	// DefaultHeartbeatInterval is the default interval between heartbeat checks.
	DefaultHeartbeatInterval = 30 * time.Minute
)

// HeartbeatConfig holds heartbeat configuration.
type HeartbeatConfig struct {
	Interval time.Duration
}

// Heartbeat runs periodic checks and triggers agent actions.
type Heartbeat struct {
	agent    *Agent
	bus      *bus.MessageBus
	interval time.Duration
}

// NewHeartbeat creates a new heartbeat for the given agent.
func NewHeartbeat(ag *Agent, mb *bus.MessageBus, interval time.Duration) *Heartbeat {
	if interval <= 0 {
		interval = DefaultHeartbeatInterval
	}
	return &Heartbeat{
		agent:    ag,
		bus:      mb,
		interval: interval,
	}
}

// Start begins the heartbeat goroutine. It blocks until ctx is cancelled.
func (hb *Heartbeat) Start(ctx context.Context) {
	slog.Info("heartbeat started",
		"agent", hb.agent.Name(),
		"interval", hb.interval,
	)

	ticker := time.NewTicker(hb.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("heartbeat stopped", "agent", hb.agent.Name())
			return
		case <-ticker.C:
			hb.tick(ctx)
		}
	}
}

func (hb *Heartbeat) tick(ctx context.Context) {
	slog.Debug("heartbeat tick (keepalive)", "agent", hb.agent.Name())

	// Heartbeat is a pure keepalive signal — it does NOT trigger LLM calls.
	// Its only job is to maintain the app's background connection and
	// periodically consolidate long-term memory from conversation logs.
	hb.updateMemory()
}

func (hb *Heartbeat) loadHeartbeatTasks() string {
	path := filepath.Join(hb.agent.workspace(), "HEARTBEAT.md")
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	content := strings.TrimSpace(string(data))
	if content == "" {
		return ""
	}
	return content
}

func (hb *Heartbeat) updateMemory() {
	slog.Info("heartbeat: triggering memory update", "agent", hb.agent.Name())
	hb.agent.memory.ReviewAndUpdateMemory(hb.agent.workspace())
}
