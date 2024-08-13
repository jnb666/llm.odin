package gpt2

import "core:os"
import "core:io"
import "core:log"

import "../array"
import "../nn"
import "../util"

Checkpoint_Magic :: 20240326
Checkpoint_Version_F32 :: 3
Checkpoint_Version_BF16 :: 5

// Create new model using config and weights from checkpoint file.
load_checkpoint :: proc($Device, $Type: typeid, file_name: string) -> (model: ^Model(Device, Type), err: os.Error) {
	log.info("load checkpoint from", file_name)
	file := os.open(file_name) or_return
	defer os.close(file)
	r := os.stream_from_handle(file)
	cfg, version := read_checkpoint_header(r) or_return
	log.debugf("version = %d\n%v", version, cfg)
	model = new_model(Device, Type, cfg)
	if version == Checkpoint_Version_BF16 {
		load_parameters(BF16, r, model) or_return
	} else {
		load_parameters(f32, r, model) or_return
	}
	return model, nil
}

// Save a checkpoint with the weights from the given model.
save_checkpoint :: proc(model: ^Model($Device, $Type), file_name: string) -> os.Error {
	log.info("save checkpoint to", file_name)
	file := os.open(file_name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644) or_return
	defer os.close(file)
	w := os.stream_from_handle(file)
	type := Checkpoint_Version_BF16 when Type == BF16 else Checkpoint_Version_F32
	write_checkpoint_header(w, model.config, i32(type)) or_return
	return save_parameters(Type, w, model)
}

read_checkpoint_header :: proc(r: io.Stream) -> (cfg: Config, version: i32, err: os.Error) {
	header := make([]i32, nn.Header_Size)
	defer delete(header)
	util.read_slice(r, header) or_return
	if header[0] != Checkpoint_Magic || (header[1] != Checkpoint_Version_F32 && header[1] != Checkpoint_Version_BF16) {
		return cfg, 0, .Invalid_File
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
	return cfg, header[1], nil
}

write_checkpoint_header :: proc(w: io.Stream, cfg: Config, version: i32) -> io.Error {
	header := make([]i32, nn.Header_Size)
	header[0] = Checkpoint_Magic
	header[1] = version
	header[2] = i32(cfg.max_seq)
	header[3] = i32(cfg.vocab_size)
	header[4] = i32(cfg.num_layers)
	header[5] = i32(cfg.num_heads)
	header[6] = i32(cfg.channels)
	header[7] = i32(cfg.vocab_padded)
	header[8] = cfg.no_bias ? 1 : 0
	return util.write_slice(w, header)
}

// set model weights from input stream
load_parameters :: proc($Data_Type: typeid, r: io.Stream, model: ^Model($D, $T)) -> io.Error {
	for n in 0 ..< 2 {
		read_parameter(Data_Type, r, &model.embed.layer, n) or_return
	}
	for l, i in model.blocks[0].layers {
		for n in 0 ..< len(l.params) {
			for block in model.blocks {
				read_parameter(Data_Type, r, block.layers[i], n) or_return
			}
		}
	}
	for n in 0 ..< 2 {
		read_parameter(Data_Type, r, &model.norm3.layer, n) or_return
	}
	return nil
}

save_parameters :: proc($Data_Type: typeid, w: io.Stream, model: ^Model($D, $T)) -> io.Error {
	for n in 0 ..< 2 {
		write_parameter(Data_Type, w, &model.embed.layer, n) or_return
	}
	for l, i in model.blocks[0].layers {
		for n in 0 ..< len(l.params) {
			for block in model.blocks {
				write_parameter(Data_Type, w, block.layers[i], n) or_return
			}
		}
	}
	for n in 0 ..< 2 {
		write_parameter(Data_Type, w, &model.norm3.layer, n) or_return
	}
	return nil
}

read_parameter :: proc($Data_Type: typeid, r: io.Stream, layer: ^nn.Layer($D, $T), n: int) -> io.Error {
	p := &layer.params[n]
	return array.read(Data_Type, r, p.arr)
}

write_parameter :: proc($Data_Type: typeid, w: io.Stream, layer: ^nn.Layer($D, $T), n: int) -> io.Error {
	p := &layer.params[n]
	return array.write(Data_Type, w, p.arr)
}

