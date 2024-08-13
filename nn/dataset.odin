package nn

import "core:io"
import "core:os"
import "core:log"
import "core:math/rand"

import "../array"
import "../util"

Dataset_Magic :: 20240520
Dataset_Version :: 1
Header_Size :: 256

// Dataset is a set of tokens for model training or validation 
Dataset :: struct($D: typeid) {
	tokens: []u16,
	inputs: Array(D, i32),
	targets: Array(D, i32),
	batch_size: int,
	seq_length: int,
	offset: int,
	shuffle: bool,
}

// Load dataset from disk and return a Dataset targetted to the given device. If shuffle flag is set then batches are selected randomly.
read_dataset :: proc($Device: typeid, file_name: string, batch_size, seq_length: int, shuffle := false) -> (d: ^Dataset(Device), err: os.Error) {
	file := os.open(file_name) or_return
	defer os.close(file)
	r := os.stream_from_handle(file)
	header := make([]i32, Header_Size)
	defer delete(header)
	util.read_slice(r, header) or_return
	if header[0] != Dataset_Magic || header[1] != Dataset_Version {
		return nil, .Invalid_File
	}
	length := int(header[2])
	file_bytes := os.seek(file, 0, os.SEEK_END) or_return
	if exp_bytes := Header_Size*size_of(i32) + length*size_of(u16); exp_bytes != int(file_bytes) {
		log.errorf("expecting file size of %d - got %d", exp_bytes, file_bytes)
		return nil, .Invalid_File
	}
	log.infof("read %d tokens from %s", length, file_name)
	_ = os.seek(file, 0, os.SEEK_SET) or_return
	d = new(Dataset(Device))
	d.tokens = make([]u16, length)
	util.read_slice(r, d.tokens) or_return
	d.batch_size = batch_size
	d.seq_length = seq_length
	d.inputs = array.zeros(Device, i32, {batch_size, seq_length})
	d.targets = array.zeros(Device, i32, {batch_size, seq_length})
	d.shuffle = shuffle
	return d, nil
}

// Write dataset to disk from slice of encoded tokens
write_dataset :: proc(file_name: string, tokens: []u16) -> os.Error {
	file := os.open(file_name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644) or_return
	defer os.close(file)
	w := os.stream_from_handle(file)
	header := make([]i32, Header_Size)
	defer delete(header)
	header[0] = Dataset_Magic
	header[1] = Dataset_Version
	header[2] = i32(len(tokens))
	log.infof("write %d tokens to %s", len(tokens), file_name)
	util.write_slice(w, header) or_return
	return util.write_slice(w, tokens)
}

// Free allocated resources for dataset
delete_dataset :: proc(d: ^Dataset($Device)) {
	delete(d.tokens)
	array.delete(d.inputs)
	array.delete(d.targets)
	free(d)
}

// Get next random batch of data - uses the current context random number generator if shuffle mode is set
// else returns false at end of data. Reset offset to 0 to rewind.
// targets are the next token in the sequence to the input
next_batch :: proc(d: ^Dataset($Device)) -> bool {
	n := d.seq_length
	if d.shuffle {
		for i in 0 ..< d.batch_size {
			pos := rand.int_max(len(d.tokens) - n - 1)
			array.copy(array.view(d.inputs, {n}, offset=i*n), d.tokens[pos:])
			array.copy(array.view(d.targets, {n}, offset=i*n), d.tokens[pos+1:])
		}
		return true
	}
	if d.offset + n*d.batch_size >= len(d.tokens) {
		return false
	}
	for i in 0 ..< d.batch_size {
		array.copy(array.view(d.inputs, {n}, offset=i*n), d.tokens[d.offset:])
		array.copy(array.view(d.targets, {n}, offset=i*n), d.tokens[d.offset+1:])
		d.offset += n
	}
	return true
}