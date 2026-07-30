package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestRunVerboseSnapshot(t *testing.T) {
	cfg := config{section: "all", steps: 1000, lr: 0.001}
	got := runForTest(t, cfg)
	assertSnapshot(t, "all_verbose.golden", got)
}

func TestRunQuietSnapshot(t *testing.T) {
	cfg := config{section: "all", steps: 1000, lr: 0.001, quiet: true}
	got := runForTest(t, cfg)
	assertSnapshot(t, "all_quiet.golden", got)
}

func TestRunJSONSnapshot(t *testing.T) {
	cfg := config{section: "all", steps: 1000, lr: 0.001, json: true}
	got := runForTest(t, cfg)
	assertSnapshot(t, "all_json.golden", got)
}

func runForTest(t *testing.T, cfg config) string {
	t.Helper()
	result := execute(cfg)
	var buf bytes.Buffer
	if err := render(&buf, cfg, result); err != nil {
		t.Fatalf("render failed: %v", err)
	}
	return buf.String()
}

func assertSnapshot(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if os.Getenv("UPDATE_SNAPSHOTS") == "1" {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatalf("write snapshot: %v", err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read snapshot %s: %v", name, err)
	}
	if string(want) != got {
		t.Fatalf("snapshot mismatch for %s\n\nwant:\n%s\n\ngot:\n%s", name, string(want), got)
	}
}
