package gpt2

import "core:encoding/json"
import "core:log"
import "core:os"
import "core:strings"

import "../array"
import "../nn"
import "../safetensors"
import "../util"

hf_wmap := map[string][]string {
	"embed" = {"wte.weight", "wpe.weight"},
	"norm3" = {"ln_f.weight", "ln_f.bias"},
}

hf_bmap := map[string][]string {
	"norm1"    = {"ln_1.weight", "ln_1.bias"},
	"pre_att"  = {"attn.c_attn.weight.T", "attn.c_attn.bias"},
	"att_proj" = {"attn.c_proj.weight.T", "attn.c_proj.bias"},
	"norm2"    = {"ln_2.weight", "ln_2.bias"},
	"ff"       = {"mlp.c_fc.weight.T", "mlp.c_fc.bias"},
	"ff_proj"  = {"mlp.c_proj.weight.T", "mlp.c_proj.bias"},
}

// Create new model using config and weights from huggingface.
load_huggingface :: proc($Device, $Type: typeid, model_name: string) -> (model: ^Model(Device, Type), err: os.Error) {
	log.infof("load %s model from huggingface hub", model_name)
	cfg := huggingface_get_config(model_name) or_return
	model = new_model(Device, Type, cfg)
	if err := huggingface_load_weights(model, model_name); err != nil {
		if e, ok := err.(os.Error); ok {
			return nil, e
		}
		log.errorf("error reading %s weights: %v", model_name, err)
		return nil, .Invalid_File
	}
	return model, nil
}

// Download model config from Huggingface hub and parse file from cache.
huggingface_get_config :: proc(model_name: string) -> (cfg: Config, err: os.Error) {
	file := util.huggingface_cache_file(model_name, "config.json") or_return
	defer delete(file)
	log.debug("get config from", file)
	Config :: struct {
		n_layer:     int,
		n_head:      int,
		n_embd:      int,
		n_positions: int,
		vocab_size:  int,
	}
	c: Config
	util.unmarshal_json_file(file, &c) or_return
	cfg = {
		num_layers = c.n_layer,
		num_heads  = c.n_head,
		channels   = c.n_embd,
		max_seq    = c.n_positions,
		vocab_size = c.vocab_size,
	}
	pad_vocab(&cfg)
	return cfg, nil
}

// Load weights from huggingface model.safetensors file
huggingface_load_weights :: proc(model: ^Model($D, $T), model_name: string) -> safetensors.Error {
	file_name := util.huggingface_cache_file(model_name, "model.safetensors") or_return
	defer delete(file_name)
	log.debug("load weights from", file_name)
	file := safetensors.open(file_name) or_return
	defer safetensors.close(file)

	cut :: proc(s: string) -> (pre, post: string) {
		ix := strings.index_byte(s, '.')
		if ix >= 0 {
			return s[:ix], s[ix + 1:]
		} else {
			return "", s
		}
	}

	for l in model.layers {
		if len(l.params) > 0 {
			assert(len(l.params) == len(hf_wmap[l.name]))
			for name, i in hf_wmap[l.name] {
				//log.debug("load", name)
				p := l.params[i].arr
				if name == "wte.weight" {
					p = array.view(p, {model.vocab_size, p.dims[1]})
				}
				safetensors.read(file, name, p) or_return
			}
		}
		for l in l.layers {
			if len(l.params) > 0 {
				ix, lname := cut(l.name)
				assert(len(l.params) == len(hf_bmap[lname]))
				for name, i in hf_bmap[lname] {
					key := strings.join({"h", ix, name}, ".")
					//log.debug("load", key)
					transpose := false
					if strings.has_suffix(key, ".T") {
						key = key[:len(key) - 2]
						transpose = true
					}
					safetensors.read(file, key, l.params[i].arr, transpose) or_return
					delete(key)
				}
			}
		}
	}
	return nil
}
