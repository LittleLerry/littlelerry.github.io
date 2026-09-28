---
title: "Getting Started with This Blog"
date: 2026-09-28 20:00:00 +0800
categories: [Blog]
tags: [jekyll, chirpy]
math: true
---

Welcome! This post exists to verify that everything renders correctly:
code highlighting, LaTeX math, and the comment section below.
Feel free to delete it once you publish a real article.

## Code blocks

Fenced code blocks get syntax highlighting and line numbers automatically:

```cuda
// A tiny GEMM accumulator, just for demo
__global__ void gemm_naive(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (int k = 0; k < N; ++k)
    acc += A[row * N + k] * B[k * N + col];
  C[row * N + col] = acc;
}
```

## Math

Inline math works with `$...$`, e.g. the element of a matrix product is
$C_{ij} = \sum_k A_{ik} B_{kj}$. Display math uses `$$...$$`:

$$
S = \sum_{k=0}^{N-1} a_k b_k, \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

## Writing posts

Create a new file in `_posts/` named `YYYY-MM-DD-slug.md` with front matter
like the one at the top of this post. Set `math: true` only when the post
uses formulas. That's it — commit, and GitHub Actions deploys the site
automatically.

## Next steps

- [ ] Replace this post with a real article
- [ ] Fill in your email and avatar in `_config.yml`
- [ ] Enable comments (see SETUP.md)
