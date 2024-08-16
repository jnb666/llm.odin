package util

import "base:intrinsics"
import "core:fmt"
import "core:io"
import "core:log"
import "core:math"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"

Cache_Dir :: "llm_odin"

user_formatters: map[typeid]fmt.User_Formatter

@(init)
init_formatters :: proc() {
	fmt.set_user_formatters(&user_formatters)
}

// Location for cache data - e.g. /home/user/.cache on linux
// allocates a new string
user_cache_dir :: proc() -> (dir: string, ok: bool) {
	when ODIN_OS == .Windows {
		if cdir, ok := os.lookup_env("LocalAppData"); ok {
			return cdir, true
		}
	} else when ODIN_OS == .Darwin {
		if home, ok := os.lookup_env("HOME"); ok {
			defer delete(home)
			return strings.concatenate({home, "/Library/Caches"}), true
		}
	} else {
		if cdir, ok := os.lookup_env("XDG_CACHE_HOME"); ok {
			return cdir, true
		}
		if home, ok := os.lookup_env("HOME"); ok {
			defer delete(home)
			return strings.concatenate({home, "/.cache"}), true
		}
	}
	return "", false
}

// Create new sequence as 0..end or start..<end range.
seq :: proc {
	seq_1,
	seq_2,
}

seq_1 :: proc($T: typeid, end: int) -> []T where intrinsics.type_is_numeric(T) {
	assert(end > 0)
	s := make([]T, end)
	for i in 0 ..< end {
		s[i] = T(i)
	}
	return s
}

seq_2 :: proc($T: typeid, start: int, end: int) -> []T where intrinsics.type_is_numeric(T) {
	assert(end > start)
	s := make([]T, end - start)
	for v, i in start ..< end {
		s[i] = T(v)
	}
	return s
}


Running_Mean :: struct {
	value: f32,
	count: f32,
}

// Add value to running mean
add_mean :: proc(m: ^Running_Mean, x: f32) {
	m.count += 1
	m.value += (x - m.value) / m.count
}


// get max absolute difference between slice elements
max_difference :: proc(a, b: []f32, loc := #caller_location) -> f32 {
	assert(len(a) == len(b), "length mismatch", loc)
	diff: f32
	for v, i in a {
		diff = max(diff, abs(v - b[i]))
	}
	return diff
}

// integer divide with round up
ceil_div :: proc "contextless" (x, y: int) -> int {
	return ((x - 1) / y) + 1
}

// floating point comparison
nearly_equal :: proc "contextless" (a, b: f32, epsilon: f32 = 1e-3, threshold: f32 = 1e-6) -> bool {
	if a == b {
		return true
	}
	diff := abs(a - b)
	norm := min(abs(a) + abs(b), math.F32_MAX)
	return diff < max(threshold, epsilon * norm)
}

// read slice of data into input stream
read_slice :: proc(r: io.Stream, data: []$T) -> io.Error {
	buf := slice.to_bytes(data)
	_, err := io.read_full(r, buf)
	return err
}

// write slice of data to output stream
write_slice :: proc(w: io.Stream, data: []$T) -> io.Error {
	buf := slice.to_bytes(data)
	_, err := io.write_full(w, buf)
	return err
}

// format number with commas on each thousand
comma_format :: proc(n: int) -> string {
	format :: proc(b: ^strings.Builder, n: int) {
		if n < 1000 {
			fmt.sbprintf(b, "%d", n)
			return
		}
		format(b, n / 1000)
		fmt.sbprintf(b, ",%03d", n % 1000)
	}

	b := strings.builder_make()
	n := n
	if n < 0 {
		strings.write_byte(&b, '-')
		n = -n
	}
	format(&b, n)
	return strings.to_string(b)
}

// Simple template parser - replaces {{var}} strings in text with fields[var]. Also replaces \n escapes with linefeeds.
parse_template :: proc(text: string, fields: map[string]string, allocator := context.allocator) -> string {
	text := text
	for key, val in fields {
		k := strings.concatenate({"{{", key, "}}"}, allocator = allocator)
		text, _ = strings.replace_all(text, k, val, allocator = allocator)
	}
	text, _ = strings.replace_all(text, "\\n", "\n", allocator = allocator)
	return text
}
