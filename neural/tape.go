// Package neural implements reverse-mode automatic differentiation using a
// computation tape. This is the standard approach for neural networks: record
// the forward pass, then replay backward to accumulate gradients. Efficient
// when dim(output) << dim(input), i.e., scalar loss with many parameters.
package neural

import "math"

// op encodes a recorded operation on the tape.
type op struct {
	kind   opKind
	args   [2]int     // indices into tape
	partials [2]float64 // local partial derivatives
}

type opKind uint8

const (
	opInput opKind = iota
	opAdd
	opMul
	opSigmoid
	opSub
)

// Tape records a computation graph for reverse-mode AD.
// The zero value is not usable; create via NewTape.
type Tape struct {
	ops  []op
	vals []float64
}

// Node is a handle to a value on the tape.
type Node int

// NewTape returns an empty tape pre-allocated for n expected nodes.
//
//go:norace
func NewTape(n int) *Tape {
	return &Tape{
		ops:  make([]op, 0, n),
		vals: make([]float64, 0, n),
	}
}

// Var records an input variable.
func (t *Tape) Var(x float64) Node {
	idx := len(t.ops)
	t.ops = append(t.ops, op{kind: opInput})
	t.vals = append(t.vals, x)
	return Node(idx)
}

// Add records a + b.
func (t *Tape) Add(a, b Node) Node {
	idx := len(t.ops)
	v := t.vals[a] + t.vals[b]
	t.ops = append(t.ops, op{kind: opAdd, args: [2]int{int(a), int(b)}, partials: [2]float64{1, 1}})
	t.vals = append(t.vals, v)
	return Node(idx)
}

// Sub records a - b.
func (t *Tape) Sub(a, b Node) Node {
	idx := len(t.ops)
	v := t.vals[a] - t.vals[b]
	t.ops = append(t.ops, op{kind: opSub, args: [2]int{int(a), int(b)}, partials: [2]float64{1, -1}})
	t.vals = append(t.vals, v)
	return Node(idx)
}

// Mul records a * b.
func (t *Tape) Mul(a, b Node) Node {
	idx := len(t.ops)
	v := t.vals[a] * t.vals[b]
	t.ops = append(t.ops, op{kind: opMul, args: [2]int{int(a), int(b)}, partials: [2]float64{t.vals[b], t.vals[a]}})
	t.vals = append(t.vals, v)
	return Node(idx)
}

// Sigmoid records σ(a) = 1/(1+exp(-a)).
func (t *Tape) Sigmoid(a Node) Node {
	idx := len(t.ops)
	s := 1.0 / (1.0 + math.Exp(-t.vals[a]))
	t.ops = append(t.ops, op{kind: opSigmoid, args: [2]int{int(a), 0}, partials: [2]float64{s * (1 - s), 0}})
	t.vals = append(t.vals, s)
	return Node(idx)
}

// Value returns the forward-pass result of a node.
func (t *Tape) Value(n Node) float64 { return t.vals[n] }

// Backward computes gradients of node `root` w.r.t. all ancestors.
// Returns a slice indexed by Node with the accumulated gradient.
//
//go:norace
func (t *Tape) Backward(root Node) []float64 {
	grad := make([]float64, len(t.ops))
	grad[root] = 1.0
	for i := int(root); i >= 0; i-- {
		if grad[i] == 0 {
			continue
		}
		o := &t.ops[i]
		switch o.kind {
		case opInput:
			// leaf
		case opAdd, opSub, opMul, opSigmoid:
			grad[o.args[0]] += grad[i] * o.partials[0]
			if o.kind != opSigmoid {
				grad[o.args[1]] += grad[i] * o.partials[1]
			}
		}
	}
	return grad
}

// ForwardSingleNeuron computes a single neuron: σ(w0*x0 + w1*x1 + b)
// and returns the output along with gradients [dL/dw0, dL/dw1, dL/db, dL/dx0, dL/dx1]
// where L = (output - target)².
func ForwardSingleNeuron(x0, x1, w0, w1, b, target float64) (output float64, grads [5]float64) {
	t := NewTape(16)
	nX0 := t.Var(x0)
	nX1 := t.Var(x1)
	nW0 := t.Var(w0)
	nW1 := t.Var(w1)
	nB := t.Var(b)
	nTarget := t.Var(target)

	// forward: σ(w0*x0 + w1*x1 + b)
	wx0 := t.Mul(nW0, nX0)
	wx1 := t.Mul(nW1, nX1)
	sum := t.Add(t.Add(wx0, wx1), nB)
	out := t.Sigmoid(sum)

	// loss: (out - target)²
	diff := t.Sub(out, nTarget)
	loss := t.Mul(diff, diff)

	output = t.Value(out)
	g := t.Backward(loss)
	grads = [5]float64{g[int(nW0)], g[int(nW1)], g[int(nB)], g[int(nX0)], g[int(nX1)]}
	return
}
