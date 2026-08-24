//! Minimal HTTP/1.1 server (std only) speaking the OpenAI SSE chat protocol.
//! One thread per connection; M3 replaces this with a batching scheduler.

use crate::backend::Net;
use crate::model::Model;
use crate::scheduler::{Event, Request, Scheduler};
use crate::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

pub struct Engine {
    pub model: Arc<Net>,
    pub tokenizer: Tokenizer,
    pub model_name: String,
    pub think: bool,
    pub scheduler: Scheduler,
    counter: AtomicU64,
}

impl Engine {
    pub fn new(model: Net, draft: Option<Model>, tokenizer: Tokenizer, model_name: String, think: bool) -> Engine {
        let model = Arc::new(model);
        let scheduler = Scheduler::start(model.clone(), draft.map(Arc::new));
        Engine { model, tokenizer, model_name, think, scheduler, counter: AtomicU64::new(1) }
    }
}

pub fn serve(addr: &str, engine: Arc<Engine>) {
    let listener = TcpListener::bind(addr).expect("bind");
    for stream in listener.incoming().flatten() {
        let engine = engine.clone();
        thread::spawn(move || handle(stream, &engine));
    }
}

fn handle(mut stream: TcpStream, engine: &Engine) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut content_length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    let _ = reader.read_exact(&mut body);

    if request_line.starts_with("GET /health") {
        respond_json(&mut stream, 200, &json!({"status": "ok", "model": engine.model_name, "device": engine.model.device()}));
        return;
    }
    if request_line.starts_with("GET /v1/models") {
        respond_json(&mut stream, 200, &json!({"object": "list", "data": [{"id": engine.model_name, "object": "model"}]}));
        return;
    }
    if !request_line.starts_with("POST /v1/chat/completions") {
        respond_json(&mut stream, 404, &json!({"error": {"message": "not found"}}));
        return;
    }
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            respond_json(&mut stream, 400, &json!({"error": {"message": format!("bad json: {e}")}}));
            return;
        }
    };
    chat(&mut stream, engine, &req);
}

/// Qwen chat template: <|im_start|>role\ncontent<|im_end|>\n … <|im_start|>assistant\n
/// Qwen3: non-thinking mode closes an empty think block. Qwen3.5 hybrids are ALWAYS-thinking:
/// their template ends with an OPEN <think>\n and the model closes it itself — feeding a closed
/// block puts the model out of distribution (garbage output).
fn build_prompt(engine: &Engine, messages: &[Value]) -> String {
    let mut p = String::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("user");
        let content = m["content"].as_str().unwrap_or("");
        p.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    p.push_str("<|im_start|>assistant\n");
    if engine.model.config().lin.is_some() {
        p.push_str("<think>\n");
    } else if !engine.think {
        p.push_str("<think>\n\n</think>\n\n");
    }
    p
}

fn chat(stream: &mut TcpStream, engine: &Engine, req: &Value) {
    let messages = req["messages"].as_array().cloned().unwrap_or_default();
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.7) as f32;
    let top_p = req["top_p"].as_f64().unwrap_or(0.9) as f32;
    let streaming = req["stream"].as_bool().unwrap_or(false);
    let stop_strs: Vec<String> = match &req["stop"] {
        Value::String(x) => vec![x.clone()],
        Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    let id = format!("chatcmpl-{}", engine.counter.fetch_add(1, Ordering::Relaxed));
    let model_name = engine.model_name.clone();

    let prompt = build_prompt(engine, &messages);
    let prompt_ids = engine.tokenizer.encode(&prompt);
    let cfg = engine.model.config();
    if prompt_ids.len() + 1 >= cfg.max_context {
        respond_json(stream, 400, &json!({"error": {"message": format!("prompt of {} tokens exceeds context {}", prompt_ids.len(), cfg.max_context)}}));
        return;
    }
    let max_tokens = max_tokens.min(cfg.max_context - prompt_ids.len() - 1);

    if streaming {
        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n");
    }
    let chunk = |content: &str, finish: Option<&str>, usage: Option<Value>| {
        chunk_delta(&id, &model_name, if content.is_empty() { json!({}) } else { json!({"content": content}) }, finish, usage)
    };
    let rchunk = |reasoning: &str| chunk_delta(&id, &model_name, json!({"reasoning": reasoning}), None, None);


    // ---- hand the request to the batching scheduler and stream its events ----
    let t0 = Instant::now();
    let tk = &engine.tokenizer;
    let (tx, rx) = std::sync::mpsc::channel();
    engine.scheduler.submit(Request {
        prompt_ids: prompt_ids.clone(),
        max_tokens,
        temperature,
        top_p,
        seed: engine.counter.load(Ordering::Relaxed).wrapping_mul(0x9E3779B97F4A7C15),
        stop_ids: vec![tk.im_end, tk.eos],
        tx,
    });
    // Qwen3.5 always-thinking models begin generation inside <think>; everything up to the
    // closing tag is reasoning, not answer content (OpenAI/OpenRouter keep them separate).
    let thinking_model = engine.model.config().lin.is_some();
    let mut in_think = thinking_model;
    let mut reasoning = String::new();
    let mut reasoning_tokens = 0usize;
    let mut prefill_s = 0.0f32;
    let mut cached = 0usize;
    let mut t_first: Option<Instant> = None;
    let mut out_ids = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut full = String::new();
    let mut finish = "length";
    let mut batch_avg = 0.0;
    let mut accept_rate = 0.0;
    for ev in rx {
        match ev {
            Event::Prefilled { seconds, cached: c } => {
                prefill_s = seconds;
                cached = c;
                t_first = Some(Instant::now());
            }
            Event::Token(next) => {
                out_ids.push(next);
                pending.extend(tk.token_bytes(next));
                // Emit the longest valid UTF-8 prefix; keep incomplete multibyte tails for the next token.
                let valid = match std::str::from_utf8(&pending) {
                    Ok(_) => pending.len(),
                    Err(e) => e.valid_up_to(),
                };
                if valid > 0 {
                    let mut text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                    pending.drain(..valid);
                    // stop sequences: cut at the first match and end the request
                    let mut hit_stop = false;
                    for st in &stop_strs {
                        let probe = format!("{full}{text}");
                        if let Some(i) = probe.find(st.as_str()) {
                            let keep = i.saturating_sub(full.len());
                            text.truncate(keep.min(text.len()));
                            hit_stop = true;
                            break;
                        }
                    }
                    if in_think {
                        // the closing tag may arrive split across tokens; buffer on `reasoning`
                        reasoning.push_str(&text);
                        reasoning_tokens += 1;
                        if let Some(i) = reasoning.find("</think>") {
                            let after = reasoning[i + "</think>".len()..].to_string();
                            reasoning.truncate(i);
                            in_think = false;
                            if streaming && !text.is_empty() {
                                write_sse(stream, &rchunk(text.trim_end()));
                            }
                            text = after.trim_start().to_string();
                            if text.is_empty() {
                                continue;
                            }
                        } else {
                            if streaming && !write_sse(stream, &rchunk(&text)) {
                                eprintln!("[{id}] client disconnected");
                                return;
                            }
                            continue;
                        }
                    }
                    full.push_str(&text);
                    if hit_stop {
                        finish = "stop";
                        if streaming && !text.is_empty() {
                            write_sse(stream, &chunk(&text, None, None));
                        }
                        break; // dropping rx retires the sequence in the scheduler
                    }
                    if streaming && !write_sse(stream, &chunk(&text, None, None)) {
                        eprintln!("[{id}] client disconnected");
                        return; // dropping rx makes the scheduler retire this sequence
                    }
                }
            }
            Event::Done { finish: f, batch_avg: b, accept_rate: a } => {
                finish = f;
                batch_avg = b;
                accept_rate = a;
                break;
            }
        }
    }
    let decode_s = t_first.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
    let _ = t0;
    eprintln!(
        "[{id}] prompt {} tok ({} cached) in {:.2}s ({:.1} tok/s prefill) | generated {} tok in {:.2}s ({:.1} tok/s decode, avg batch {:.1}, draft accept {:.0}%)",
        prompt_ids.len(), cached, prefill_s, (prompt_ids.len() - cached) as f32 / prefill_s.max(1e-6),
        out_ids.len(), decode_s, out_ids.len() as f32 / decode_s.max(1e-6), batch_avg, accept_rate * 100.0
    );

    let usage = json!({"prompt_tokens": prompt_ids.len(), "completion_tokens": out_ids.len(), "total_tokens": prompt_ids.len() + out_ids.len(),
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "cached_tokens": cached, "prefill_tok_per_sec": (prompt_ids.len() - cached) as f32 / prefill_s.max(1e-6), "decode_tok_per_sec": out_ids.len() as f32 / decode_s.max(1e-6)});
    if streaming {
        write_sse(stream, &chunk("", Some(finish), Some(usage)));
        let _ = write!(stream, "{:x}\r\ndata: [DONE]\n\n\r\n0\r\n\r\n", "data: [DONE]\n\n".len());
    } else {
        let mut msg = json!({"role": "assistant", "content": full});
        if !reasoning.trim().is_empty() {
            msg["reasoning"] = json!(reasoning.trim());
        }
        respond_json(stream, 200, &json!({"id": id, "object": "chat.completion", "model": model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}));
    }
}

fn chunk_delta(id: &str, model: &str, delta: Value, finish: Option<&str>, usage: Option<Value>) -> Value {
    let mut c = json!({"id": id, "object": "chat.completion.chunk", "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    if let Some(u) = usage {
        c["usage"] = u;
    }
    c
}

fn write_sse(stream: &mut TcpStream, v: &Value) -> bool {
    let payload = format!("data: {v}\n\n");
    write!(stream, "{:x}\r\n{}\r\n", payload.len(), payload).is_ok() && stream.flush().is_ok()
}

fn respond_json(stream: &mut TcpStream, code: u16, v: &Value) {
    let body = v.to_string();
    let status = match code { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", _ => "Error" };
    let _ = write!(stream, "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
}
