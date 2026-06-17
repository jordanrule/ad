// Command demo exercises the three AD implementations and compares results.
package main

import (
	"fmt"

	"github.com/jrule/ad/neural"
	"github.com/jrule/ad/optim"
	"github.com/jrule/ad/particle"
)

func main() {
	fmt.Println("=== AD Done Three Ways ===")
	fmt.Println()

	// 1. Forward-mode dual numbers — particle detector response
	fmt.Println("1) Particle Detector (forward-mode, dual numbers)")
	fmt.Println("   response(E) = sqrt(E)*sin(E) + exp(-E/10)")
	for _, e := range []float64{1.0, 5.0, 20.0} {
		val, deriv := particle.SimulateDetectorResponse(e)
		fmt.Printf("   E=%5.1f  →  response=%.6f  d(response)/dE=%.6f\n", e, val, deriv)
	}
	fmt.Println()

	// 2. Reverse-mode tape — single neuron backprop
	fmt.Println("2) Neural Network (reverse-mode, tape)")
	fmt.Println("   σ(w0*x0 + w1*x1 + b), loss = (out - target)²")
	out, grads := neural.ForwardSingleNeuron(
		1.0, 2.0, // x0, x1
		0.5, -0.3, // w0, w1
		0.1,       // bias
		0.8,       // target
	)
	fmt.Printf("   output=%.6f\n", out)
	fmt.Printf("   ∂L/∂w0=%.6f  ∂L/∂w1=%.6f  ∂L/∂b=%.6f\n", grads[0], grads[1], grads[2])
	fmt.Printf("   ∂L/∂x0=%.6f  ∂L/∂x1=%.6f\n", grads[3], grads[4])
	fmt.Println()

	// 3. Graph-free reverse-mode — Rosenbrock optimization
	fmt.Println("3) General Optimization (reverse-mode, topological sort)")
	fmt.Println("   Minimizing Rosenbrock: f(x,y) = (1-x)² + 100*(y-x²)²")
	x, y := optim.MinimizeRosenbrock(-1.0, 1.0, 0.001, 10000)
	fmt.Printf("   After 10000 steps: x=%.6f  y=%.6f  (optimum at 1,1)\n", x, y)
}
