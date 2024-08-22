package gpt2

import "core:bytes"
import "core:log"
import "core:slice"
import "core:testing"

import "../nn"
import "../util"


@(test)
huggingface_test :: proc(t: ^testing.T) {
	cfg, err := huggingface_get_config("gpt2")
	log.debug(cfg)
	testing.expect_value(t, err, nil)

	model := new_model(CPU, f32, cfg)
	defer delete_model(model)

	err2 := huggingface_load_weights(model, "gpt2")
	testing.expect_value(t, err2, nil)
}

@(test)
load_tokenizer_test :: proc(t: ^testing.T) {
	tok := new_tokenizer()
	defer delete_tokenizer(tok)

	log.debugf("got %d entries in vocab", vocab_size(tok))
	log.debugf("got %d entries in merges", len(tok.bpe_ranks))
	testing.expect_value(t, vocab_size(tok), 50257)
	testing.expect_value(t, len(tok.bpe_ranks), 50000)
	testing.expect_value(t, tok.encoder[End_Token], 50256)
}

@(test)
encode_test :: proc(t: ^testing.T) {
	tok := tokenizer()
	defer nn.delete_tokenizer(&tok)

	test_string := "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗<|endoftext|>"
	expect_tokens := []u16{15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245, 50256}

	tokens := nn.encode(&tok, test_string)
	defer delete(tokens)
	log.debug("tokens:", tokens)
	testing.expect(t, slice.equal(tokens, expect_tokens))

	text := nn.decode(&tok, ..tokens)
	defer delete(text)
	log.debug("decoded:", text)
	testing.expect_value(t, text, test_string)
}


@(test)
init_test :: proc(t: ^testing.T) {
	cfg := Config {
		num_layers = 12,
		num_heads  = 12,
		channels   = 768,
		max_seq    = 1024,
		vocab_size = 50257,
	}
	pad_vocab(&cfg)
	log.debug(cfg)

	model := new_model(CPU, f32, cfg)
	defer delete_model(model)

	init_weights(model)
	log.debug("embed wte", model.embed.wte.arr)
}

@(test)
summary_test :: proc(t: ^testing.T) {
	cfg := Config {
		num_layers = 12,
		num_heads  = 12,
		channels   = 768,
		max_seq    = 1024,
		vocab_size = 50257,
	}
	pad_vocab(&cfg)
	log.debug(cfg)

	model := new_model(CPU, f32, cfg)
	defer delete_model(model)
	build(model, 4, 256)

	buf: bytes.Buffer
	defer bytes.buffer_destroy(&buf)
	nn.write_summary(bytes.buffer_to_stream(&buf), &model.layer)
	summary := bytes.buffer_to_string(&buf)
	log.debugf("\n%s", summary)

	testing.expect_value(
		t,
		summary,
		`┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                      == GPT2_124M - device: CPU, data type: f32 ==                      │
├───────────────────────────────────────────────────────┬─────────────────┬───────────────┤
│  Layer                                                │  Output shape   │  Parameters   │
├───────────────────────────────────────────────────────┼─────────────────┼───────────────┤
│  embed{ wte:[50304, 768] wpe:[1024, 768] } (Encoder)  │  [4 256 768]    │   39,419,904  │
│  block0{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block1{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block2{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block3{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block4{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block5{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block6{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block7{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block8{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block9{ heads:12 channels:768 } (Transformer)        │  [4 256 768]    │    7,087,872  │
│  block10{ heads:12 channels:768 } (Transformer)       │  [4 256 768]    │    7,087,872  │
│  block11{ heads:12 channels:768 } (Transformer)       │  [4 256 768]    │    7,087,872  │
│  norm3{ channels:768 } (Layernorm)                    │  [4 256 768]    │        1,536  │
│  rev_embed{ nin:768 nout:50304 } (Linear)             │  [4 256 50304]  │            0  │
│                                                       │                 │       ──────  │
│                                                       │                 │  124,475,904  │
└───────────────────────────────────────────────────────┴─────────────────┴───────────────┘
`,
	)
}
