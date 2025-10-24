// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package web

import (
	"iter"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2aclient/agentcard"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/restapi/services"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func getFreePort(t *testing.T) int {
	t.Helper()
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("net.ResolveTCPAddr() error = %v", err)
	}
	listener, err := net.ListenTCP("tcp", addr)
	if err != nil {
		t.Fatalf("net.ListenTCP() error = %v", err)
	}
	defer func() {
		if err := listener.Close(); err != nil {
			t.Fatalf("listener.Close() error = %v", err)
		}
	}()
	tcpAddr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("listener.Addr() = %T, want net.TCPAddr", listener.Addr())
	}
	return tcpAddr.Port
}

func TestWebLauncher_ServesA2A(t *testing.T) {
	ctx := t.Context()

	port := getFreePort(t)

	launcher, _, err := BuildLauncher([]string{"--serve_a2a", "--port", strconv.Itoa(port)})
	if err != nil {
		t.Fatalf("BuildLauncher() error = %v", err)
	}

	wantMessage := "Hello, world!"
	agnt, err := agent.New(agent.Config{
		Name: "HelloWorldAgent",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ic.InvocationID())
				event.Content = genai.NewContentFromText(wantMessage, genai.RoleModel)
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	config := &adk.Config{
		AgentLoader:    services.NewSingleAgentLoader(agnt),
		SessionService: session.InMemoryService(),
	}

	go func() {
		if err := launcher.Run(t.Context(), config); err != nil {
			t.Errorf("launcher.Run() error = %v", err)
		}
	}()

	var card *a2a.AgentCard
	for retry := range 3 {
		time.Sleep(20 * time.Millisecond)
		cardResolver := agentcard.Resolver{BaseURL: "http://127.0.0.1:" + strconv.Itoa(port)}
		card, err = cardResolver.Resolve(ctx)
		if err == nil {
			break
		}
		if retry == 2 {
			t.Fatalf("cardResolver.Resolve() error = %v", err)
		}
	}

	insecureGRPC := a2aclient.WithGRPCTransport(
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	client, err := a2aclient.NewFromCard(ctx, card, insecureGRPC)
	if err != nil {
		t.Fatalf("a2aclient.NewFromCard() error = %v", err)
	}

	got, err := client.SendMessage(ctx, &a2a.MessageSendParams{
		Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "Hi!"}),
	})
	if err != nil {
		t.Fatalf("client.SendMessage() error = %v", err)
	}
	task, ok := got.(*a2a.Task)
	if !ok {
		t.Fatalf("client.SendMessage() result type = %T, want a2a.Task", got)
	}
	if len(task.Artifacts) != 1 {
		t.Fatalf("len(task.Artifacts) = %d, want 1", len(task.Artifacts))
	}
	parts := task.Artifacts[0].Parts
	if len(parts) != 1 {
		t.Fatalf("len(task.Artifacts[0].Parts) = %d, want 1", len(parts))
	}
	if gotPart, ok := parts[0].(a2a.TextPart); !ok || gotPart.Text != wantMessage {
		t.Fatalf("task.Artifacts[0].Parts[0] = %v, want %v", parts[0], a2a.TextPart{Text: wantMessage})
	}
}
