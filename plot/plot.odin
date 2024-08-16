package plot

import "core:encoding/json"
import "core:fmt"
import "core:log"
import "core:mem"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:time"

import "../util"

Max_Line_Length :: 100

// Webview plotting context
Context :: struct {
	thread:       ^thread.Thread,
	webview:      Webview,
	width:        int,
	height:       int,
	border:       int,
	table_header: int,
	log_level:    log.Level,
	log_opts:     log.Options,
	loaded:       bool,
}

// Stats represents a series of traces which are rendered using plotly
Stats :: struct {
	traces:  [dynamic]Trace,
	samples: [dynamic]Sample,
}

// One plot trace is a series of x,y points and associated metadata
Trace :: struct {
	x:       [dynamic]i32,
	y:       [dynamic]f32,
	name:    string,
	line:    struct {
		width: int,
	},
	opacity: f64,
}

// Sample is a text generation sample
Sample :: struct {
	step:    int,
	preview: string,
	text:    string,
}

// Create new thread to open a webview window
start :: proc(width, height: int, border := 10, table_header := 20, xrange: []int = nil) -> ^Context {
	c := new(Context)
	c.width = width
	c.height = height
	c.border = border
	c.table_header = table_header
	c.log_level = context.logger.lowest_level
	c.log_opts = context.logger.options
	c.thread = thread.create_and_start_with_poly_data(c, webview_thread)
	// ensure JS initialization has completed
	for !sync.atomic_load(&c.loaded) {
		time.sleep(10 * time.Millisecond)
	}
	if xrange != nil && len(xrange) >= 1 {
		if len(xrange) == 1 {
			set_xrange(c, 0, xrange[0])
		} else {
			set_xrange(c, xrange[0], xrange[1])
		}
	}
	return c
}

// Wait for webview to exit and free allocated data
wait :: proc(c: ^Context) {
	if c.thread != nil {
		log.debug("wait for window to close")
		thread.destroy(c.thread)
	}
	free(c)
}

// Delete all allocated objects
delete_stats :: proc(s: ^Stats) {
	for trace in s.traces {
		delete(trace.x)
		delete(trace.y)
	}
	delete(s.traces)
	for sample in s.samples {
		delete(sample.text)
	}
	delete(s.samples)
}

// Set x axis range xmin to xmax
set_xrange :: proc(c: ^Context, xmin, xmax: int) -> Webview_Error {
	update := fmt.aprintf("setXRange(%d, %d)\x00", xmin, xmax)
	defer delete(update)
	log.debug(update)
	return webview_eval(c.webview, strings.unsafe_string_to_cstring(update))
}

// Draw the plot traces 
update_plot :: proc(c: ^Context, stats: ^Stats) -> Webview_Error {
	return call_js(c, "updatePlot", stats.traces)
}

// Refresh the table of samples
update_table :: proc(c: ^Context, stats: ^Stats) -> Webview_Error {
	return call_js(c, "updateTable", stats.samples)
}

call_js :: proc(c: ^Context, function: string, value: $T) -> Webview_Error {
	data, err := json.marshal(value)
	if err != nil {
		log.panic("json marshal error:", err)
	}
	defer delete(data)
	update := strings.concatenate({function, "(", string(data), ")", "\x00"})
	defer delete(update)
	//log.debug(update)
	return webview_eval(c.webview, strings.unsafe_string_to_cstring(update))
}

// Add point to named trace at x, y and set attributes on first point
add :: proc(s: ^Stats, name: string, x: int, y: f32, width := 2, opacity := 1.0) {
	for &t in s.traces {
		if t.name == name {
			append(&t.x, i32(x))
			append(&t.y, y)
			return
		}
	}
	append(&s.traces, Trace{x = {i32(x)}, y = {y}, name = name, line = {width = width}, opacity = opacity})
}

// Add text sample - makes a copy of the input text
add_sample :: proc(s: ^Stats, step: int, text: string) {
	text := strings.clone(text)
	end := len(text)
	if n := strings.index_byte(text, '\n'); n > 0 {
		end = n
	}
	end = min(end, Max_Line_Length)
	append(&s.samples, Sample{step = step, preview = text[:end], text = text})
}

webview_thread :: proc(c: ^Context) {
	context.logger = log.create_console_logger(c.log_level, c.log_opts)
	w := webview_create(1, nil)
	if w == nil {
		log.panic("error creating webview")
	}
	webview_set_title(w, "Odin llm plot")
	webview_set_size(w, i32(c.width), i32(c.height), .NONE)
	height := f64(c.height - 2 * c.border)
	content := get_html(c.width - 2 * c.border, int(height * 0.6), int(height * 0.4) - c.table_header, c.table_header)
	must(webview_set_html(w, content))
	delete(content)
	// called from webview JS once initialization is done
	loaded_callback :: proc "cdecl" (seq, req: cstring, arg: rawptr) {
		c := cast(^Context)arg
		sync.atomic_store(&c.loaded, true)
	}
	must(webview_bind(w, "plotLoaded", loaded_callback, c))
	c.webview = w
	log.debug("webview loaded")
	must(webview_run(w))
	log.debug("webview closed")
}

get_html :: proc(width, plot_height, table_height, table_header: int) -> cstring {
	plot_template := #load("plot.html", string)
	fields := map[string]string {
		"width"        = fmt.aprint(width, allocator = context.temp_allocator),
		"plot_height"  = fmt.aprint(plot_height, allocator = context.temp_allocator),
		"table_height" = fmt.aprint(table_height, allocator = context.temp_allocator),
		"table_header" = fmt.aprint(table_header, allocator = context.temp_allocator),
	}
	defer mem.free_all(context.temp_allocator)
	defer delete(fields)
	html := util.parse_template(plot_template, fields, allocator = context.temp_allocator)
	return strings.clone_to_cstring(html)
}
