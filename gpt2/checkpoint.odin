package gpt2

import "core:os"
import "core:io"
import "core:log"

import "../array"
import "../nn"
import "../util"

Checkpoint_Magic :: 20240326
Checkpoint_Version :: 3

// Create new model using config and weights from checkpoint file.
load_checkpoint :: proc($Device, $Type: typeid, file_name: string) -> (model: ^Model(Device, Type), err: os.Error) {
	log.info("load checkpoint from", file_name)
	file := os.open(file_name) or_return
	defer os.close(file)

	r := os.stream_from_handle(file)
	cfg := read_checkpoint_header(r) or_return
	log.debugf("\n%v", cfg)

	model = new_model(Device, Type, cfg)
	load_parameters(r, model) or_return
	return model, nil
}

read_checkpoint_header :: proc(r: io.Stream) -> (cfg: Config, err: os.Error) {
	header := make([]i32, 256)
	defer delete(header)
	util.read_slice(r, header) or_return
	if header[0] != Checkpoint_Magic || header[1] != Checkpoint_Version {
		return cfg, .Invalid_File
	}
	cfg = Config{
		max_seq 	= int(header[2]),
		vocab_size 	= int(header[3]),
		num_layers	= int(header[4]),
		num_heads	= int(header[5]),
		channels	= int(header[6]),
		vocab_padded= int(header[7]),
		no_bias		= header[8] != 0,
	}
	return cfg, nil
}

// set model weights from input stream
load_parameters :: proc(r: io.Stream, model: ^Model($D, $T)) -> io.Error {
	for n in 0 ..< 2 {
		read_parameter(r, &model.embed.layer, n) or_return
	}
	for l, i in model.blocks[0].layers {
		for n in 0 ..< len(l.params) {
			for block in model.blocks {
				read_parameter(r, block.layers[i], n) or_return
			}
		}
	}
	for n in 0 ..< 2 {
		read_parameter(r, &model.norm3.layer, n) or_return
	}
	return nil
}

read_parameter :: proc(r: io.Stream, layer: ^nn.Layer($D, $T), n: int) -> io.Error {
	p := &layer.params[n]
	return array.read(f32, r, p.arr)
}

