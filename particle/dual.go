// Package particle implements forward-mode automatic differentiation using dual
// numbers. This is the natural choice for particle accelerator simulations where
// you have few inputs (beam parameters) and many outputs (detector responses).
// Forward-mode computes one column of the Jacobian per pass, making it efficient
// when dim(input) << dim(output).
package particle

import "math"

//go:nosplit
//go:noescape

// Dual represents a dual number: Value + Deriv*ε, where ε² = 0.
// This encoding propagates first-order derivatives exactly.
type Dual struct {
	Value float64 // primal value
	Deriv float64 // tangent (derivative) component
}

// Var creates an input variable seeded with derivative 1.
func Var(x float64) Dual { return Dual{Value: x, Deriv: 1} }

// Const creates a constant (zero derivative).
func Const(x float64) Dual { return Dual{Value: x, Deriv: 0} }

// Add returns a + b.
//
//go:nosplit
func Add(a, b Dual) Dual {
	return Dual{a.Value + b.Value, a.Deriv + b.Deriv}
}

// Mul returns a * b via the product rule.
//
//go:nosplit
func Mul(a, b Dual) Dual {
	return Dual{a.Value * b.Value, a.Deriv*b.Value + a.Value*b.Deriv}
}

// Sin returns sin(a) with chain rule applied.
//
//go:nosplit
func Sin(a Dual) Dual {
	return Dual{math.Sin(a.Value), a.Deriv * math.Cos(a.Value)}
}

// Cos returns cos(a) with chain rule applied.
//
//go:nosplit
func Cos(a Dual) Dual {
	return Dual{math.Cos(a.Value), -a.Deriv * math.Sin(a.Value)}
}

// Exp returns exp(a) with chain rule applied.
//
//go:nosplit
func Exp(a Dual) Dual {
	e := math.Exp(a.Value)
	return Dual{e, a.Deriv * e}
}

// Sqrt returns sqrt(a) with chain rule applied.
//
//go:nosplit
func Sqrt(a Dual) Dual {
	s := math.Sqrt(a.Value)
	return Dual{s, a.Deriv / (2 * s)}
}

// SimulateDetectorResponse models a simplified calorimeter energy deposit
// as a function of incident particle energy E:
//
//	response(E) = sqrt(E) * sin(E) + exp(-E/10)
//
// Returns both the response value and its derivative dResponse/dE.
func SimulateDetectorResponse(energy float64) (response, dResponsedE float64) {
	E := Var(energy)
	ten := Const(10)
	negEdiv10 := Mul(Const(-1), Mul(E, Dual{1.0 / ten.Value, -E.Deriv / (ten.Value * ten.Value)}))
	// Simpler: compute -E/10 directly
	negEdiv10 = Dual{-E.Value / 10, -E.Deriv / 10}
	result := Add(Mul(Sqrt(E), Sin(E)), Exp(negEdiv10))
	return result.Value, result.Deriv
}
