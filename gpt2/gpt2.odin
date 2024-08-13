package gpt2

import "core:fmt"
import "core:mem"
import "core:log"
import "core:os"
import "core:math"
import "core:strings"

import "../nn"
import "../array"
import "../util"

Array :: array.Array
BF16 :: array.BF16
CPU :: array.CPU
Cuda :: array.Cuda

// GPT2 configuration 
Config :: struct {
	num_layers: int,
	num_heads: int,
	channels: int,
	max_seq: int,
	vocab_size: int,
	vocab_padded: int,
	no_bias: bool,
	recompute_gelu: bool,
}

// Pad size of vocabulary - used for embedding wte parameter and logits array
pad_vocab :: proc(cfg: ^Config, align := 128) {
	cfg.vocab_padded = mem.align_forward_int(cfg.vocab_size, align)
}

// GPT2 model definition
Model :: struct($D, $T: typeid) {
	using config: Config,
	using layer: nn.Layer(D, T),
	embed: nn.Encoder(D, T),
	blocks: []^Transformer(D, T),
	norm3: nn.Layernorm(D, T),
	rev_embed: nn.Linear(D, T),
	act: Activations(D, T),
}

Activations :: struct($D, $T: typeid) {
	encoded: Array(D, T),
	feed_forward: []Array(D, T),
	final_norm: Array(D, T),
	logits: Array(D, T),
	losses: Array(D, f32),
	have_dlogits: bool,
}

// Initialise model and allocate layers and associated parameters which are set to zero
new_model :: proc($D, $T: typeid, cfg: Config) -> ^Model(D, T) {
	m := new(Model(D, T))
	m.config = cfg
	m.type = "Model"
	m.layers = make([]^nn.Layer(D, T), cfg.num_layers + 3)

	m.embed = nn.make_encoder(D, T, "embed", cfg.vocab_padded, cfg.max_seq, cfg.channels)
	m.layers[0] = &m.embed.layer
	m.blocks = make([]^Transformer(D, T), cfg.num_layers)
	for i in 0 ..< cfg.num_layers {
		m.blocks[i] = make_transformer(D, T, &m.config, i)
		m.layers[i+1] = &m.blocks[i].layer
	}
	m.norm3 = nn.make_layernorm(D, T, "norm3", cfg.channels)
	m.layers[cfg.num_layers+1] = &m.norm3.layer
	m.rev_embed = nn.make_linear(D, T, "rev_embed", cfg.channels, cfg.vocab_padded, no_init=true)
	m.rev_embed.weight = m.embed.wte
	m.layers[cfg.num_layers+2] = &m.rev_embed.layer

	for layer in m.layers {
		m.num_params += layer.num_params
	}
	m.name = fmt.aprintf("GPT2_%dM", m.num_params / 1e6)
	m.info = fmt.aprintf("GPT2_%v", cfg)
	m.out_shape = array.make_shape({0, 0, cfg.vocab_padded})
	return m
}

// Initialise random model weights using the current context random source
init_weights :: proc(m: ^Model($Dev, $Typ), weight_scale: f32 = 0.02) {

	init_layer :: proc(l: ^nn.Layer(Dev, Typ), weight_scale, proj_scale: f32, vocab_size: int) {
		if len(l.params) > 0 {
			switch l.type {
			case "Layernorm":
				array.fill(l.params[0].arr, 1)
				array.zero(l.params[1].arr)
			case "Linear":
				sigma := strings.has_suffix(l.name, "proj") ? weight_scale/proj_scale : weight_scale
				nn.normal_init(l.params[0].arr, stddev=sigma)
				array.zero(l.params[1].arr)
			case "Encoder":
				// for wte - note that weights may be padded - only init the ones that are used
				array.zero(l.params[0].arr)
				view := array.view(l.params[0].arr, {vocab_size, l.params[0].dims[1]})
				nn.normal_init(view, stddev=weight_scale)
				nn.normal_init(l.params[1].arr, stddev=weight_scale)
			case:
				log.panic("invalid layer type", l.type)
			}
		}
		for layer in l.layers {
			init_layer(layer, weight_scale, proj_scale, vocab_size)
		}
	}

	proj_scale := 1 / math.sqrt(1 / f32(2*m.num_layers))
	log.debugf("initialise GPT2 weights: stddev=%.4g proj_stddev=%.4g", weight_scale, weight_scale/proj_scale)
	for layer in m.layers {
		init_layer(layer, weight_scale, proj_scale, m.vocab_size)
	}
}

// Forward propagate input tokens through model to generate logits with prediction of next token. 
forward :: proc(m: ^Model($Dev, $Typ), inp: Array(Dev, i32), train := true, loc := #caller_location) {
	assert(inp.ndims == 2 && inp.dims[1] <= m.max_seq, "invalid input shape", loc)
	B, T := inp.dims[0], inp.dims[1]
	if m.act.encoded.dims[0] != B || m.act.encoded.dims[1] != T {
		build(m, B, T)
	}
	nn.encoder_forward(&m.embed, inp, m.act.encoded)
	x := m.act.encoded
	for block, i in m.blocks {
		transformer_forward(block, x, m.act.feed_forward[i], train)
		x = m.act.feed_forward[i]
	}
	nn.layernorm_forward(&m.norm3, x, m.act.final_norm, train)
	if m.act.logits.ndims != 3 || m.act.logits.dims[0] != B || m.act.logits.dims[1] != T {
		array.delete(m.act.logits)
		m.act.logits = array.zeros(Dev, Typ, {B, T, m.vocab_padded})
	}
	nn.linear_forward(&m.rev_embed, m.act.final_norm, m.act.logits)
	m.act.have_dlogits = false
}

// Calculate cross entropy loss from logits populated from forward step and target predictions. Returns mean loss over the batch.
// If train flag is set then will overwrite the logit activations with the logit gradients as first backprop step.
calc_loss :: proc(m: ^Model($Dev, $Typ), targets: Array(Dev, i32), train := true, grad_accum_steps := 1, loc := #caller_location) -> f32 {
	nn.cross_entropy_loss(m.act.logits, targets, m.act.losses, m.vocab_size, train, grad_accum_steps, loc=loc)
	m.act.have_dlogits = train
	return array.mean(m.act.losses)
}

// Backward propagate from the loss to update the gradients of all of the trainable parameters.
// Gradients are accumulated - call nn.zero_gradients before each step to reset before each batch.
// calc_loss should be called first to to initialise the logit gradients.
backward :: proc(m: ^Model($Dev, $Typ), inp: Array(Dev, i32), loc := #caller_location) {
	assert(m.act.have_dlogits, "must call calc_loss as first step of back prop", loc)
	assert(inp.dims == m.act.losses.dims, "invalid input shape", loc)
	tmp1 := array.zeros_like(m.act.encoded)
	defer array.delete(tmp1)
	nn.linear_backward(&m.rev_embed, m.act.final_norm, m.act.logits, tmp1)
	array.delete(m.act.logits)
	m.act.logits = {}
	tmp2 := m.act.final_norm
	array.zero(tmp2)
	nn.layernorm_backward(&m.norm3, m.act.feed_forward[m.num_layers-1], tmp1, tmp2)
	for i := m.num_layers-1; i >= 0; i -= 1 {
		input := i > 0 ? m.act.feed_forward[i-1] : m.act.encoded
		transformer_backward(m.blocks[i], input, tmp2, tmp1)
		tmp1, tmp2 = tmp2, tmp1
	}
	nn.encoder_backward(&m.embed, inp, tmp2)
}

// Generate output tokens from the model using the given sampler and initial prompt tokens.
// Will exit after either max_length tokens are generated or the stop_token is received.
// Calls fn after generating each token
generate :: proc(m: ^Model($Dev, $Typ), sampler: nn.Sampler, ctx: rawptr, fn: proc(ctx: rawptr, token: u16, done: bool), prompt: []u16, 
										max_length := 256, stop_token := -1, loc := #caller_location) {
	assert(len(prompt) < m.max_seq, "prompt is too long", loc)
	assert(max_length <= m.max_seq, "max length exceeds model sequence length", loc)

	tokens := make([]u16, max_length)
	defer delete(tokens)
	copy(tokens, prompt)

	input := array.zeros(Dev, i32, {1, max_length})
	defer array.delete(input)
	logits := make([]f32, m.vocab_size)
	defer delete(logits)

	t := len(prompt)
	done := false
	for !done {
		array.copy(input, tokens)
		forward(m, input, train=false)
		output := array.view(m.act.logits, {m.vocab_size}, offset=(t-1)*m.vocab_padded)
		array.copy(logits, output)
		token := nn.sample(sampler, logits)
		tokens[t] = token
		t += 1
		done = t >= max_length || (stop_token >= 0 && int(token) == stop_token)
		fn(ctx, token, done)
	}
}

// Set model output shapes and allocate activation arrays
build :: proc(m: ^Model($Dev, $Typ), input_shape: ..int, loc := #caller_location) {
	assert(len(input_shape) == 2, "invalid input shape", loc)
	B, T, C := input_shape[0], input_shape[1], m.channels
	nn.build(&m.layer, B, T)
	log.debugf("build activations - input shape = %v", input_shape)
	delete_activations(&m.act)
	m.act.encoded = array.zeros(Dev, Typ, {B, T, C})
	m.act.feed_forward = make([]Array(Dev, Typ), m.num_layers)
	for i in 0 ..< m.num_layers {
		m.act.feed_forward[i] = array.zeros(Dev, Typ, {B, T, C})
	}
	m.act.final_norm = array.zeros(Dev, Typ, {B, T, C})
	m.act.losses = array.zeros(Dev, f32, {B, T})
}

// set parameter gradients to zero. will allocate them if not already done
zero_gradients :: proc(m: ^Model($D, $T)) {
	for &layer in m.layers {
		nn.zero_gradients(layer)
	}
}

delete_activations :: proc(act: ^Activations($D, $T)) {
	array.delete(act.encoded)
	for i in 0 ..< len(act.feed_forward) {
		array.delete(act.feed_forward[i])
	}
	delete(act.feed_forward)
	array.delete(act.final_norm)
	array.delete(act.losses)
}

delete_model :: proc(m: ^Model($D, $T)) {
	delete_activations(&m.act)
	nn.delete_layer(m.embed)
	for block in m.blocks {
		delete_transformer(block)
	}
	delete(m.blocks)
	nn.delete_layer(m.norm3)
	nn.delete_layer(m.rev_embed)
	nn.delete_layer_base(m.layer)
	free(m)
}

// Transformer block with multi-head self attention.
Transformer :: struct($D, $T: typeid) {
	using config: ^Config,
	using layer: nn.Layer(D, T),
	norm1: nn.Layernorm(D, T),
	pre_att: nn.Linear(D, T),
	attn: nn.Attention(D, T),
	att_proj: nn.Linear(D, T),
	norm2: nn.Layernorm(D, T),
	ff: nn.Linear(D, T),
	ff_proj: nn.Linear(D, T),
	act: Transformer_Activations(D, T),
}

Transformer_Activations :: struct($D, $T: typeid) {
	norm1: Array(D, T),
	qkv: Array(D, T),
	att: Array(D, T),
	res: Array(D, T),
	norm2: Array(D, T),
	ff_out: Array(D, T),
	ff_gelu: Array(D, T),
}

make_transformer :: proc($D, $T: typeid, cfg: ^Config, layer: int) -> ^Transformer(D, T) {
	name :: proc(b: []u8, ix: int, s: string) -> string {
		return fmt.bprintf(b, "%d.%s", ix, s)
	}
	buf: [128]u8
	bias := !cfg.no_bias
	t := new(Transformer(D, T))
	t.config = cfg
	t.norm1 = nn.make_layernorm(D, T, name(buf[:], layer, "norm1"), cfg.channels)
	t.pre_att = nn.make_linear(D, T, name(buf[:], layer, "pre_att"), cfg.channels, 3*cfg.channels, bias=bias)
	t.attn = nn.make_attention(D, T, name(buf[:], layer, "attn"), cfg.num_heads, cfg.channels)
	t.att_proj = nn.make_linear(D, T, name(buf[:], layer, "att_proj"), cfg.channels, cfg.channels, bias=bias)
	t.norm2 = nn.make_layernorm(D, T, name(buf[:], layer, "norm2"), cfg.channels)
	t.ff = nn.make_linear(D, T, name(buf[:], layer, "ff"), cfg.channels, 4*cfg.channels, bias=bias)
	t.ff_proj = nn.make_linear(D, T, name(buf[:], layer, "ff_proj"), 4*cfg.channels, cfg.channels, bias=bias)

	t.type = "Transformer"
	t.name = fmt.aprintf("block%d", layer)
	t.info = fmt.aprintf("%s{{ heads:%d channels:%d }}", t.name, t.attn.num_heads, t.attn.channels)
	t.out_shape = array.make_shape({0, 0, cfg.channels})
	t.layers = make([]^nn.Layer(D, T), 7)
	copy(t.layers, []^nn.Layer(D, T){&t.norm1.layer, &t.pre_att.layer, &t.attn.layer, &t.att_proj.layer, &t.norm2.layer, &t.ff.layer, &t.ff_proj.layer})
	for layer in t.layers {
		t.num_params += layer.num_params
	}
	return t
}

transformer_forward :: proc(t: ^Transformer($Dev, $Typ), inp, out: Array(Dev, Typ), train := true) {
	B, T, C := inp.dims[0], inp.dims[1], t.attn.channels
	if t.act.norm1.dims[0] != B || t.act.norm1.dims[1] != T {
		build_transformer(t, B, T)
	}
	tmp := array.zeros(Dev, Typ, {B, T, C})
	defer array.delete(tmp)
	nn.layernorm_forward(&t.norm1, inp, t.act.norm1, train)
	nn.linear_forward(&t.pre_att, t.act.norm1, t.act.qkv)
	nn.attention_forward(&t.attn, t.act.qkv, t.act.att)
	nn.linear_forward(&t.att_proj, t.act.att, tmp)
	nn.add(inp, tmp, t.act.res)
	nn.layernorm_forward(&t.norm2, t.act.res, t.act.norm2, train)
	nn.linear_forward(&t.ff, t.act.norm2, t.act.ff_out)
	ff_gelu := t.act.ff_gelu
	if t.recompute_gelu {
		ff_gelu = array.zeros(Dev, Typ, {B, T, 4*C})
	}
	nn.gelu_forward(t.act.ff_out, ff_gelu)
	nn.linear_forward(&t.ff_proj, ff_gelu, tmp)
	if t.recompute_gelu {
		array.delete(ff_gelu)
	}
	nn.add(t.act.res, tmp, out)
}

transformer_backward :: proc(t: ^Transformer($Dev, $Typ), inp, dout, din: Array(Dev, Typ), loc := #caller_location) {
	B, T, C := inp.dims[0], inp.dims[1], t.attn.channels
	tmp1 := array.zeros(Dev, Typ, {B, T, 4*C})
	defer array.delete(tmp1)
	array.copy(din, dout)
	ff_gelu := t.act.ff_gelu
	if t.recompute_gelu {
		ff_gelu = array.zeros(Dev, Typ, {B, T, 4*C})
		nn.gelu_forward(t.act.ff_out, ff_gelu)
	}
	nn.linear_backward(&t.ff_proj, ff_gelu, dout, tmp1)
	if t.recompute_gelu {
		array.delete(ff_gelu)
	}
	nn.gelu_backward(t.act.ff_out, tmp1, tmp1)
	tmp2 := dout
	nn.linear_backward(&t.ff, t.act.norm2, tmp1, tmp2)
	nn.layernorm_backward(&t.norm2, t.act.res, tmp2, din)
	nn.linear_backward(&t.att_proj, t.act.att, din, tmp2)
	tmp3 := array.view(tmp1, {B, T, 3*C})
	nn.attention_backward(&t.attn, t.act.qkv, tmp2, tmp3)
	nn.linear_backward(&t.pre_att, t.act.norm1, tmp3, tmp2)
	nn.layernorm_backward(&t.norm1, inp, tmp2, din)
}

build_transformer :: proc(t: ^Transformer($Dev, $Typ), B, T: int) {
	delete_transformer_activations(&t.act)
	C := t.attn.channels
	t.act.norm1 = array.zeros(Dev, Typ, {B, T, C})
	t.act.qkv = array.zeros(Dev, Typ, {B, T, 3*C})
	t.act.att = array.zeros(Dev, Typ, {B, T, C})
	t.act.res = array.zeros(Dev, Typ, {B, T, C})
	t.act.norm2 = array.zeros(Dev, Typ, {B, T, C})
	t.act.ff_out = array.zeros(Dev, Typ, {B, T, 4*C})
	if !t.recompute_gelu {
		t.act.ff_gelu = array.zeros(Dev, Typ, {B, T, 4*C})
	}
}

delete_transformer_activations :: proc(a: ^Transformer_Activations($D, $T)) {
	array.delete(a.norm1)
	array.delete(a.qkv)
	array.delete(a.att)
	array.delete(a.res)
	array.delete(a.norm2)
	array.delete(a.ff_out)
	array.delete(a.ff_gelu)
}

delete_transformer :: proc(t: ^Transformer($D, $T)) {
	delete_transformer_activations(&t.act)
	nn.delete_layer(t.norm1)
	nn.delete_layer(t.pre_att)
	nn.delete_layer(t.attn)
	nn.delete_layer(t.att_proj)
	nn.delete_layer(t.norm2)
	nn.delete_layer(t.ff)
	nn.delete_layer(t.ff_proj)
	nn.delete_layer_base(t.layer)
	free(t)
}
