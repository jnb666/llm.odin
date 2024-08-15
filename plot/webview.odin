package plot

import "core:log"

// bindings for https://github.com/webview/webview
//
// first build the shared library from that repo and install copy libwebview.so to /usr/local/lib
foreign import "system:webview"

Webview :: distinct rawptr

Webview_Version :: struct {
	major: u32,
	minor: u32,
	patch: u32,
}

Webview_Version_Info :: struct {
	version:        Webview_Version,
	version_number: [32]u8,
	pre_release:    [48]u8,
	build_metadata: [48]u8,
}

Webview_Hint :: enum i32 {
	NONE,
	MIN,
	MAX,
	FIXED,
}

Webview_Error :: enum i32 {
	MISSING_DEPENDENCY = -5,
	CANCELED           = -4,
	INVALID_STATE      = -3,
	INVALID_ARGUMENT   = -2,
	UNSPECIFIED        = -1,
	OK                 = 0,
	DUPLICATE          = 1,
	NOT_FOUND          = 2,
}

@(default_calling_convention = "c")
foreign webview {
	webview_create :: proc(debug: i32, window: rawptr) -> Webview ---
	webview_destroy :: proc(w: Webview) -> Webview_Error ---
	webview_run :: proc(w: Webview) -> Webview_Error ---
	webview_terminate :: proc(w: Webview) -> Webview_Error ---
	webview_dispatch :: proc(w: Webview, fn: #type proc(w: Webview, arg: rawptr), arg: rawptr) -> Webview_Error ---
	webview_get_window :: proc(w: Webview) -> rawptr ---
	webview_set_title :: proc(w: Webview, title: cstring) -> Webview_Error ---
	webview_set_size :: proc(w: Webview, width, height: i32, hints: Webview_Hint) -> Webview_Error ---
	webview_navigate :: proc(w: Webview, url: cstring) -> Webview_Error ---
	webview_set_html :: proc(w: Webview, html: cstring) -> Webview_Error ---
	webview_init :: proc(w: Webview, js: cstring) -> Webview_Error ---
	webview_eval :: proc(w: Webview, js: cstring) -> Webview_Error ---
	webview_bind :: proc(w: Webview, name: cstring, fn: #type proc(seq, req: cstring, arg: rawptr), arg: rawptr) -> Webview_Error ---
	webview_unbind :: proc(w: Webview, name: cstring) -> Webview_Error ---
	webview_ret :: proc(w: Webview, seq: cstring, status: i32, result: cstring) -> Webview_Error ---
	webview_version :: proc() -> ^Webview_Version_Info ---
}

must :: proc(e: Webview_Error, loc := #caller_location) {
	if int(e) < 0 {
		log.panicf("webview error: %v\n", e, location = loc)
	}
}
