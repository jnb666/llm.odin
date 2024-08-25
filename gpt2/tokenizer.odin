package gpt2

import "core:bytes"
import "core:encoding/json"
import "core:log"
import "core:slice"
import "core:strings"
import "core:unicode"
import "core:unicode/utf8"

import "../nn"

End_Token :: "<|endoftext|>"

End_Token_ID :: 50256

// Implement nn.Tokenizer interface
tokenizer :: proc() -> nn.Tokenizer(u16) {
	tok := new_tokenizer()
	t: nn.Tokenizer(u16)
	t.state = tok
	t.encode = proc(state: rawptr, text: string, tokens: ^[dynamic]u16) {
		encode_to(cast(^Tokenizer)state, text, tokens)
	}
	t.decode = proc(state: rawptr, tokens: []u16) -> string {
		return decode(cast(^Tokenizer)state, tokens)
	}
	t.delete = proc(state: rawptr) {
		delete_tokenizer(cast(^Tokenizer)state)
	}
	t.vocab_size = vocab_size(tok)
	t.end_token = End_Token_ID
	return t
}


// Tokenizer to convert between text and integer tokens using GPT-2 byte pair encoding
Tokenizer :: struct {
	byte_enc:  [256]rune,
	byte_dec:  [512]u8,
	encoder:   map[string]u16,
	decoder:   map[u16]string,
	bpe_ranks: map[Pair]int,
	cache:     map[string][]u16,
}

Pair :: [2]string

// GPT2 tokenizer with default vocabulary and merge files
new_tokenizer :: proc() -> ^Tokenizer {
	vocab_data := #load("encoder.json")
	json_data, err := json.parse(vocab_data)
	assert(err == nil, "error parsing encoder.json")
	defer json.destroy_value(json_data)
	vocab := make(map[string]u16)
	defer delete(vocab)
	for name, token in json_data.(json.Object) {
		vocab[name] = u16(token.(f64))
	}
	merge_data := #load("vocab.bpe", string)
	pairs: [dynamic]Pair
	for line in strings.split_lines_iterator(&merge_data) {
		if len(pairs) == 0 && strings.has_prefix(line, "#version:") {
			continue
		}
		ix := strings.index_byte(line, ' ')
		assert(ix >= 1, "error parsing vocab.bpe")
		append(&pairs, Pair{line[:ix], line[ix + 1:]})
	}
	defer delete(pairs)
	return new_tokenizer_with_config(vocab, pairs[:])
}


// Initialize new tokenizer with given vocabulary and merge list. Makes a copy of the input data.
new_tokenizer_with_config :: proc(vocab: map[string]u16, merges: []Pair) -> ^Tokenizer {
	t := new(Tokenizer)
	n := 256
	for i in 0 ..< 256 {
		if is_printable_byte(u8(i)) {
			t.byte_enc[i] = rune(i)
		} else {
			t.byte_enc[i] = rune(n)
			n += 1
		}
	}
	for v, i in t.byte_enc {
		t.byte_dec[int(v)] = u8(i)
	}
	t.encoder = make(map[string]u16)
	t.decoder = make(map[u16]string)
	for key, token in vocab {
		name := strings.clone(key)
		t.encoder[name] = token
		t.decoder[token] = name
	}
	t.bpe_ranks = make(map[Pair]int)
	for p, i in merges {
		pair := Pair{strings.clone(p[0]), strings.clone(p[1])}
		t.bpe_ranks[pair] = i
	}
	t.cache = make(map[string][]u16)
	return t
}

// Number of words in vocabulary
vocab_size :: proc(t: ^Tokenizer) -> int {
	return len(t.encoder)
}

// Clean up all allocated memory
delete_tokenizer :: proc(t: ^Tokenizer) {
	for name in t.encoder {
		delete(name)
	}
	for pair in t.bpe_ranks {
		delete(pair[0])
		delete(pair[1])
	}
	delete(t.encoder)
	delete(t.decoder)
	delete(t.bpe_ranks)
	for k, v in t.cache {
		delete(k)
		delete(v)
	}
	delete(t.cache)
	free(t)
}

// Encode text string to list of tokens using byte pair encoding.
encode_to :: proc(t: ^Tokenizer, text: string, tokens: ^[dynamic]u16) {
	parts := strings.split(text, End_Token)
	defer delete(parts)
	matches: [dynamic]int
	for part, i in parts {
		if i > 0 {
			append(tokens, t.encoder[End_Token])
		}
		runes := utf8.string_to_runes(part)
		split_words(runes, &matches)
		start := 0
		for end in matches {
			word := utf8.runes_to_string(runes[start:end])
			encode_word(t, word, tokens)
			delete(word)
			start = end
		}
		delete(runes)
	}
	delete(matches)
}

// encode a single word to one or more tokens and append to tokens array
encode_word :: proc(t: ^Tokenizer, text: string, tokens: ^[dynamic]u16) {
	item := encode_bytes(t, text)
	if toks, ok := t.cache[item]; ok {
		//log.debug("cached", item)
		append(tokens, ..toks)
		delete(item)
		return
	}
	//log.debug("lookup", item)
	word := split_runes(item)
	defer delete(word)

	pairs: [dynamic]Pair
	defer delete(pairs)
	get_pairs(word[:], &pairs)
	if len(pairs) == 0 {
		append(tokens, ..encode_item(t, item, word[:]))
		return
	}

	defer free_all(context.temp_allocator)
	new_word: [dynamic]string
	defer delete(new_word)

	for {
		first, second, found := min_rank(t, pairs[:])
		if !found {
			break
		}
		clear(&new_word)
		i := 0
		for i < len(word) {
			if j := index(word[:], first, i); j >= 0 {
				append(&new_word, ..word[i:j])
				i = j
			} else {
				append(&new_word, ..word[i:])
				break
			}
			if word[i] == first && i < len(word) - 1 && word[i + 1] == second {
				append(&new_word, strings.concatenate({first, second}, context.temp_allocator))
				i += 2
			} else {
				append(&new_word, word[i])
				i += 1
			}
		}
		word, new_word = new_word, word
		if len(word) == 1 {
			break
		}
		get_pairs(word[:], &pairs)
	}
	append(tokens, ..encode_item(t, item, word[:]))
}

// Decode tokens back to original text string.
decode :: proc(t: ^Tokenizer, tokens: []u16) -> string {
	b: bytes.Buffer
	for tok in tokens {
		item, ok := t.decoder[tok]
		if !ok {
			log.panicf("decode error: token %d not found", tok)
		}
		for ch in item {
			bytes.buffer_write_byte(&b, t.byte_dec[ch])
		}
	}
	return bytes.buffer_to_string(&b)
}

// Below are all internal private functions
@(private)
min_rank :: proc(t: ^Tokenizer, pairs: []Pair) -> (first, second: string, ok: bool) {
	INT_MAX :: int(1e10)
	best := INT_MAX
	bigram: Pair
	for p in pairs {
		if rank, rok := t.bpe_ranks[p]; rok && rank < best {
			best = rank
			bigram = p
		}
	}
	return bigram[0], bigram[1], best < INT_MAX
}

@(private)
encode_item :: proc(t: ^Tokenizer, item: string, word: []string) -> []u16 {
	tokens := make([]u16, len(word))
	for s, i in word {
		ok: bool
		if tokens[i], ok = t.encoder[s]; !ok {
			log.panicf("encode error: %s not found", s)
		}
	}
	t.cache[item] = tokens
	return tokens
}

// get all unique pairs of consecutive elements in word list
@(private)
get_pairs :: proc(word: []string, pairs: ^[dynamic]Pair) {
	clear(pairs)
	prev := word[0]
	for s in word[1:] {
		entry := Pair{prev, s}
		if !slice.contains(pairs[:], entry) {
			append(pairs, entry)
		}
		prev = s
	}
}

@(private)
split_runes :: proc(s: string) -> (word: [dynamic]string) {
	s := s
	n := utf8.rune_count_in_string(s)
	for _ in 0 ..< n {
		_, w := utf8.decode_rune_in_string(s)
		append(&word, s[:w])
		s = s[w:]
	}
	return word
}

// Index of e in s where index must be >= start or -1 if not found
@(private)
index :: proc(s: []string, e: string, start: int) -> int {
	for i := start; i < len(s); i += 1 {
		if s[i] == e {
			return i
		}
	}
	return -1
}

// apply byte encoding to string - each byte is a printable rune so as to fake valid utf-8
@(private)
encode_bytes :: proc(t: ^Tokenizer, word: string) -> string {
	b: strings.Builder
	for i in 0 ..< len(word) {
		r := t.byte_enc[word[i]]
		strings.write_rune(&b, r)
	}
	return strings.to_string(b)
}

// Word split should be equivalent to this regexp: 
//    's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
@(private)
split_words :: proc(runes: []rune, matches: ^[dynamic]int) {
	pos := 0
	clear(matches)
	match_funcs := []proc(_: rune) -> bool{unicode.is_letter, unicode.is_number, not_space_letter_or_number}

	main_loop: for pos < len(runes) {
		r := runes[pos]
		if r == '\'' && pos < len(runes) - 1 {
			if n := match_quote_suffix(runes[pos + 1:]); n > 0 {
				pos += n + 1
				append(matches, pos)
				continue main_loop
			}
		}
		for fn in match_funcs {
			if n, ok := count_rune_is(runes[pos:], fn, leading_space = true); ok {
				pos += n
				append(matches, pos)
				continue main_loop
			}
		}
		if n, ok := count_rune_is(runes[pos:], is_ascii_space); ok {
			pos += n
			append(matches, pos)
		} else {
			pos += 1
		}
	}
}

// helpers
@(private)
is_printable_byte :: proc(b: u8) -> bool {
	return (b >= '!' && b <= '~') || (b >= '¡' && b <= '¬') || (b >= '®' && b <= 'ÿ')
}

@(private)
is_ascii_space :: proc(r: rune) -> bool {
	return r == ' ' || r == '\r' || r == '\n' || r == '\t' || r == '\f' || r == '\v'
}

@(private)
not_space_letter_or_number :: proc(r: rune) -> bool {
	return !is_ascii_space(r) && !unicode.is_letter(r) && !unicode.is_number(r)
}

// returns no. of runes matched + flag indicating any not including leading space
@(private)
count_rune_is :: proc(runes: []rune, fn: proc(_: rune) -> bool, leading_space := false) -> (int, bool) {
	n := 0
	ok := false
	for r, i in runes {
		if leading_space && i == 0 && r == ' ' {
			n += 1
		} else if fn(r) {
			n += 1
			ok = true
		} else {
			break
		}
	}
	return n, ok
}

@(private)
match_quote_suffix :: proc(r: []rune) -> int {
	if len(r) >= 1 && (r[0] == 's' || r[0] == 't' || r[0] == 'm' || r[0] == 'd') {
		return 1
	}
	if len(r) >= 2 && (r[0] == 'r' && r[1] == 'e') || (r[0] == 'v' && r[1] == 'e') || (r[0] == 'l' && r[1] == 'l') {
		return 2
	}
	return 0
}
