// Package optim implements reverse-mode automatic differentiation using operator
// overloading with an implicit DAG. Unlike the tape approach, nodes here carry
// their own backward closures, and gradients are computed via a topological sort
// over the live subgraph. This is suited for general-purpose optimization where
// the computation structure varies between iterations (e.g., L-BFGS, constraint
// satisfaction).
package optim

import (
	"math"
	"sync/atomic"
)

// Value represents a differentiable scalar in the computation DAG.
// The backward closure propagates gradient to children without heap-allocating
// a separate tape structure.
type Value struct {
	data     float64
	grad     float64
	backward func()
	children [2]*Value
	mark     uint32
}

var backwardStamp uint32

// New creates a leaf Value (no parents).
//
//go:nosplit
func New(x float64) *Value {
	return &Value{data: x}
}

// Data returns the scalar value.
func (v *Value) Data() float64 { return v.data }

// Grad returns the accumulated gradient after Backward().
func (v *Value) Grad() float64 { return v.grad }

// ZeroGrad resets the gradient.
func (v *Value) ZeroGrad() { v.grad = 0 }

// Add returns a + b.
func Add(a, b *Value) *Value {
	out := &Value{data: a.data + b.data, children: [2]*Value{a, b}}
	out.backward = func() {
		a.grad += out.grad
		b.grad += out.grad
	}
	return out
}

// Mul returns a * b.
func Mul(a, b *Value) *Value {
	out := &Value{data: a.data * b.data, children: [2]*Value{a, b}}
	out.backward = func() {
		a.grad += b.data * out.grad
		b.grad += a.data * out.grad
	}
	return out
}

// Pow returns a^n (n constant).
func Pow(a *Value, n float64) *Value {
	if n == 0 {
		return New(1)
	}
	if n == 1 {
		return a
	}
	if n == 2 {
		out := &Value{data: a.data * a.data, children: [2]*Value{a, nil}}
		out.backward = func() {
			a.grad += 2 * a.data * out.grad
		}
		return out
	}
	p := math.Pow(a.data, n)
	dp := n * math.Pow(a.data, n-1)
	out := &Value{data: p, children: [2]*Value{a, nil}}
	out.backward = func() {
		a.grad += dp * out.grad
	}
	return out
}

// Neg returns -a.
func Neg(a *Value) *Value {
	out := &Value{data: -a.data, children: [2]*Value{a, nil}}
	out.backward = func() {
		a.grad -= out.grad
	}
	return out
}

// Sub returns a - b.
func Sub(a, b *Value) *Value {
	out := &Value{data: a.data - b.data, children: [2]*Value{a, b}}
	out.backward = func() {
		a.grad += out.grad
		b.grad -= out.grad
	}
	return out
}

// Exp returns exp(a).
func Exp(a *Value) *Value {
	e := math.Exp(a.data)
	out := &Value{data: e, children: [2]*Value{a, nil}}
	out.backward = func() {
		a.grad += e * out.grad
	}
	return out
}

// Backward performs reverse-mode AD from this node via topological sort.
// No external tape is needed; the DAG is walked from the root.
//
//go:norace
func (v *Value) Backward() {
	order := make([]*Value, 0, 32)
	stamp := atomic.AddUint32(&backwardStamp, 1)
	var topo func(*Value)
	topo = func(n *Value) {
		if n == nil {
			return
		}
		if n.mark == stamp {
			return
		}
		n.mark = stamp
		topo(n.children[0])
		topo(n.children[1])
		order = append(order, n)
	}
	topo(v)
	v.grad = 1.0
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}

// MinimizeRosenbrock demonstrates gradient descent on the Rosenbrock function:
//
//	f(x,y) = (1-x)² + 100*(y-x²)²
//
// Returns (x, y) after n steps with learning rate lr.
func MinimizeRosenbrock(x0, y0, lr float64, steps int) (float64, float64) {
	x := New(x0)
	y := New(y0)
	one := New(1)
	hundred := New(100)
	for range steps {
		x.ZeroGrad()
		y.ZeroGrad()
		// f = (1-x)² + 100*(y-x²)²
		oneMinusX := Sub(one, x)
		xSq := Pow(x, 2)
		yMinusXSq := Sub(y, xSq)
		f := Add(Pow(oneMinusX, 2), Mul(hundred, Pow(yMinusXSq, 2)))
		f.Backward()
		x.data -= lr * x.grad
		y.data -= lr * y.grad
	}
	return x.data, y.data
}
