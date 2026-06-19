# Automatic Differentiation Done Three Ways in Go

The most rewarding open source moment of my career was receiving a design review on automatic differentiation code I submitted to [Sebastien Binet](https://github.com/sbinet) at [CERN](https://home.cern/). I spent years building security software in Go while most of my day-to-day work focused on operations research and machine learning infrastructure, and that combination shaped this project. This repository distills what I learned into three practical approaches to AD in Go, released under an MIT license.

## Three Approaches

| # | Package | Domain | AD Mode | Key Insight |
|---|---------|--------|---------|-------------|
| 1 | `particle` | Physics particle accelerator | Forward-mode (dual numbers) | Propagate derivatives alongside values through a simulation — natural for few-input, many-output systems like detector response functions. |
| 2 | `neural` | Neural networks | Reverse-mode (tape-based) | Record a computation graph, then backpropagate — efficient when outputs are few (loss) but parameters are many (weights). |
| 3 | `optim` | General optimization | Reverse-mode (operator overloading, graph-free) | Minimal allocations via a topological sort on a DAG built implicitly — suited for arbitrary objective functions. |

## When to Use Each Approach

**1. Forward-mode (dual numbers) — `particle`**

Imagine you turn one knob (beam energy) and watch a hundred dials respond (detector readings). Forward-mode answers: "if I nudge this one input, how does every output change?" It computes all those sensitivities in a single pass. Use it when you have a small number of inputs but many outputs — sensor simulations, control systems, pricing a single instrument across many scenarios.

**2. Reverse-mode with a tape — `neural`**

Now imagine the opposite: you have a million knobs (network weights) but only care about one dial (the loss). Reverse-mode records everything that happened during the forward pass onto a "tape," then plays the tape backward to figure out how each knob contributed to the final answer. This is backpropagation — the engine behind training neural networks. Use it when you have many inputs but a single (or few) outputs to optimize.

**3. Reverse-mode with topological sort — `optim`**

Same reverse idea, but instead of a flat tape you let the computation build a tree of relationships on the fly, then walk that tree in reverse order. This is more flexible: the shape of your computation can change every iteration (think "if-else" inside your objective function). Use it for general-purpose optimization — Rosenbrock, constrained problems, or any objective whose structure isn't fixed.

**Rule of thumb:** few inputs → forward-mode. Few outputs → reverse-mode. Dynamic computation → graph-based reverse-mode.

## Usage

```bash
go run ./cmd/demo
```

## License

MIT
