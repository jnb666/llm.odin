package nn

import "core:os"
import "core:log"
import "core:math/rand"

import "../array"
import "../util"

Dataset_Magic :: 20240520
Dataset_Version :: 1


// Dataset is a set of tokens for model training or validation 
Dataset :: struct($D: typeid) {
	batch_size: int,
	seq_length: int,
	tokens: []u16,
	inputs: Array(D, i32),
	targets: Array(D, i32),
}

// Load dataset from disk and return a Dataset targetted to the given device
read_dataset :: proc($Device: typeid, file_name: string, batch_size, seq_length: int) -> (d: ^Dataset(Device), err: os.Error) {
	file := os.open(file_name) or_return
	defer os.close(file)

	r := os.stream_from_handle(file)
	header := make([]i32, 256)
	defer delete(header)

	util.read_slice(r, header) or_return
	if header[0] != Dataset_Magic || header[1] != Dataset_Version {
		return nil, .Invalid_File
	}
	length := int(header[2])
	log.infof("read %d tokens from %s", length, file_name)

	d = new(Dataset(Device))
	d.tokens = make([]u16, length)
	util.read_slice(r, d.tokens) or_return
	d.batch_size = batch_size
	d.seq_length = seq_length
	d.inputs = array.zeros(Device, i32, {batch_size, seq_length})
	d.targets = array.zeros(Device, i32, {batch_size, seq_length})
	return d, nil
}

// Free allocated resources for dataset
delete_dataset :: proc(d: ^Dataset($Device)) {
	delete(d.tokens)
	array.delete(d.inputs)
	array.delete(d.targets)
	free(d)
}

// Get next random batch of data - uses the current context random number generator
// targets are the next token in the sequence to the input
next_batch :: proc(d: ^Dataset($Device)) {
	n := d.seq_length
	for i in 0 ..< d.batch_size {
		pos := rand.int_max(len(d.tokens) - n - 1)
		array.copy(array.view(d.inputs, {n}, offset=i*n), d.tokens[pos:])
		array.copy(array.view(d.targets, {n}, offset=i*n), d.tokens[pos+1:])
	}
}