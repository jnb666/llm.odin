package nn

import "core:math"
import "core:math/rand"
import "core:slice"

// Sampler is used to sample the probability outputs to get the next token
Sampler :: struct {
	temperature: f32,
	top_k: int,
	top_p: f32,
}

Sample_Index :: struct {
	p: f32,
	i: u16,
}

// Get next token from predicted logits
sample :: proc(s: Sampler, logits: []f32) -> u16 {
	if s.temperature == 0 {
		return sample_greedy(logits)
	}
	n := len(logits)
	inp := make([]f32, n)
	defer delete(inp)
	for v, i in logits {
		inp[i] = v / s.temperature
	}
	probs := softmax(inp)
	defer delete(probs)
	if s.top_p > 0 && s.top_p < 1 {
		return sample_top_p(probs, s.top_p)
	} else if s.top_k > 0 && s.top_k < len(logits) {
		return sample_top_k(probs, s.top_k)
	} else {
		return sample_random(probs)
	}
}

// Return index of logit with highest probability
sample_greedy :: proc(logits: []f32) -> u16 {
	max_p := logits[0]
	max_i := 0
	for p, i in logits[1:] {
		if p > max_p {
			max_i, max_p = i+1, p
		}
	}
	return u16(max_i)
}

// Select random index from logits, weighted by probability
sample_random :: proc(probs: []f32) -> u16 {
	coin := rand.float32()
	cdf: f32
	for prob, i in probs {
		cdf += prob
		if coin < cdf {
			return u16(i)
		}
	}
	return u16(len(probs) - 1)
}

// Randomly select token, but limited to top_k probability entries
sample_top_k :: proc(probs: []f32, top_k: int) -> u16 {
	index := make([]Sample_Index, len(probs))
	defer delete(index)
	for p, i in probs {
		index[i] = Sample_Index{p, u16(i)}
	}
	slice.sort_by(index, proc(i, j: Sample_Index) -> bool{ return j.p < i.p })
	total_p: f32
	for ix in index[:top_k] {
		total_p += ix.p
	}
	coin := rand.float32() * total_p
	cdf: f32
	for ix in index[:top_k] {
		cdf += ix.p
		if coin < cdf {
			return ix.i
		}
	}
	return index[top_k-1].i
}

// Select tokens from the smallest subset of probabilities that sum to greater than p
//  i.e. cutoff based on probabilty rather than number of tokens
sample_top_p :: proc(probs: []f32, top_p: f32) -> u16 {
	cutoff := (1 - top_p) / f32(len(probs)-1)
	index:  [dynamic]Sample_Index
	defer delete(index)
	for p, i in probs {
		if p >= cutoff {
			append(&index, Sample_Index{p, u16(i)})
		}
	}
	slice.sort_by(index[:], proc(i, j: Sample_Index) -> bool{ return j.p < i.p })
	total_p: f32
	last_index := len(index) - 1
	for ix, i in index {
		total_p += ix.p
		if total_p > top_p {
			last_index = i
			break
		}
	}
	coin := rand.float32() * total_p
	cdf: f32
	for ix in index {
		cdf += ix.p
		if coin < cdf {
			return ix.i
		}
	}
	return index[last_index].i
}

softmax :: proc(x: []f32) -> []f32 {
	probs := make([]f32, len(x))
	maxval := slice.max(x)
	sum: f32
	for v, i in x {
		probs[i] = math.exp(v - maxval)
		sum += probs[i]
	}
	for i in 0 ..< len(x) {
		probs[i] /= sum
	}
	return probs
}