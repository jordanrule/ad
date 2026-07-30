// Command demo runs the three AD implementations with a small, readable CLI.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/jrule/ad/neural"
	"github.com/jrule/ad/optim"
	"github.com/jrule/ad/particle"
)

type config struct {
	section string
	steps   int
	lr      float64
	json    bool
	quiet   bool
}

type demoResult struct {
	Section  string           `json:"section"`
	Particle []particleResult `json:"particle,omitempty"`
	Neural   *neuralResult    `json:"neural,omitempty"`
	Optim    *optimResult     `json:"optim,omitempty"`
}

type particleResult struct {
	Energy      float64 `json:"energy"`
	Response    float64 `json:"response"`
	DResponsedE float64 `json:"d_response_dE"`
}

type neuralResult struct {
	Output float64    `json:"output"`
	Grads  [5]float64 `json:"grads"`
}

type optimResult struct {
	Steps int     `json:"steps"`
	LR    float64 `json:"lr"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
}

func main() {
	cfg := parseFlags()
	if err := validateConfig(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(2)
	}
	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "run error: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags() config {
	cfg := config{}
	flag.StringVar(&cfg.section, "section", "all", "demo section: all|particle|neural|optim")
	flag.IntVar(&cfg.steps, "steps", 10000, "gradient descent steps for the optim section")
	flag.Float64Var(&cfg.lr, "lr", 0.001, "learning rate for the optim section")
	flag.BoolVar(&cfg.json, "json", false, "output results as JSON")
	flag.BoolVar(&cfg.quiet, "quiet", false, "suppress explanatory text for quick comparisons")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [flags]\n\n", os.Args[0])
		fmt.Fprintln(flag.CommandLine.Output(), "Automatic differentiation demos in Go.")
		fmt.Fprintln(flag.CommandLine.Output())
		flag.PrintDefaults()
	}
	flag.Parse()
	cfg.section = strings.ToLower(strings.TrimSpace(cfg.section))
	return cfg
}

func validateConfig(cfg config) error {
	switch cfg.section {
	case "all", "particle", "neural", "optim":
	default:
		return fmt.Errorf("unknown section %q (expected all|particle|neural|optim)", cfg.section)
	}
	if cfg.steps <= 0 {
		return errors.New("steps must be > 0")
	}
	if cfg.lr <= 0 {
		return errors.New("lr must be > 0")
	}
	return nil
}

func run(cfg config) error {
	result := execute(cfg)
	return render(os.Stdout, cfg, result)
}

func execute(cfg config) demoResult {
	result := demoResult{Section: cfg.section}
	if cfg.section == "all" || cfg.section == "particle" {
		result.Particle = computeParticle()
	}
	if cfg.section == "all" || cfg.section == "neural" {
		result.Neural = computeNeural()
	}
	if cfg.section == "all" || cfg.section == "optim" {
		result.Optim = computeOptim(cfg)
	}
	return result
}

func computeParticle() []particleResult {
	out := make([]particleResult, 0, 3)
	for _, e := range []float64{1.0, 5.0, 20.0} {
		val, deriv := particle.SimulateDetectorResponse(e)
		out = append(out, particleResult{Energy: e, Response: val, DResponsedE: deriv})
	}
	return out
}

func computeNeural() *neuralResult {
	out, grads := neural.ForwardSingleNeuron(
		1.0, 2.0,
		0.5, -0.3,
		0.1,
		0.8,
	)
	return &neuralResult{Output: out, Grads: grads}
}

func computeOptim(cfg config) *optimResult {
	x, y := optim.MinimizeRosenbrock(-1.0, 1.0, cfg.lr, cfg.steps)
	return &optimResult{Steps: cfg.steps, LR: cfg.lr, X: x, Y: y}
}

func render(w io.Writer, cfg config, result demoResult) error {
	if cfg.json {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}
	if cfg.quiet {
		renderQuiet(w, result)
		return nil
	}
	renderVerbose(w, result)
	return nil
}

func renderVerbose(w io.Writer, result demoResult) {
	fmt.Fprintln(w, "=== AD Done Three Ways ===")
	fmt.Fprintf(w, "section=%s\n\n", result.Section)

	if len(result.Particle) > 0 {
		fmt.Fprintln(w, "1) Particle Detector (forward-mode, dual numbers)")
		fmt.Fprintln(w, "   response(E) = sqrt(E) * sin(E) + exp(-E/10)")
		for _, p := range result.Particle {
			fmt.Fprintf(w, "   E=%5.1f  response=% .6f  dResponse/dE=% .6f\n", p.Energy, p.Response, p.DResponsedE)
		}
		fmt.Fprintln(w)
	}

	if result.Neural != nil {
		fmt.Fprintln(w, "2) Neural Network (reverse-mode, tape)")
		fmt.Fprintln(w, "   out = sigmoid(w0*x0 + w1*x1 + b), loss = (out - target)^2")
		fmt.Fprintf(w, "   output=% .6f\n", result.Neural.Output)
		fmt.Fprintf(w, "   dL/dw0=% .6f  dL/dw1=% .6f  dL/db=% .6f\n", result.Neural.Grads[0], result.Neural.Grads[1], result.Neural.Grads[2])
		fmt.Fprintf(w, "   dL/dx0=% .6f  dL/dx1=% .6f\n", result.Neural.Grads[3], result.Neural.Grads[4])
		fmt.Fprintln(w)
	}

	if result.Optim != nil {
		fmt.Fprintln(w, "3) General Optimization (reverse-mode, topological sort)")
		fmt.Fprintln(w, "   f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
		fmt.Fprintf(w, "   after %d steps (lr=%g): x=% .6f  y=% .6f  optimum=(1,1)\n", result.Optim.Steps, result.Optim.LR, result.Optim.X, result.Optim.Y)
	}
}

func renderQuiet(w io.Writer, result demoResult) {
	for _, p := range result.Particle {
		fmt.Fprintf(w, "particle E=%.1f response=% .6f dResponse/dE=% .6f\n", p.Energy, p.Response, p.DResponsedE)
	}
	if result.Neural != nil {
		fmt.Fprintf(w, "neural output=% .6f dL/dw0=% .6f dL/dw1=% .6f dL/db=% .6f dL/dx0=% .6f dL/dx1=% .6f\n",
			result.Neural.Output,
			result.Neural.Grads[0], result.Neural.Grads[1], result.Neural.Grads[2],
			result.Neural.Grads[3], result.Neural.Grads[4],
		)
	}
	if result.Optim != nil {
		fmt.Fprintf(w, "optim steps=%d lr=%g x=% .6f y=% .6f\n", result.Optim.Steps, result.Optim.LR, result.Optim.X, result.Optim.Y)
	}
}
