//! Continuous-batching scheduler. One thread owns the model; HTTP handlers submit requests
//! and receive token events over channels. Each decode step runs *all* active sequences
//! through the weights together, so N concurrent users cost ~one weight pass, not N.

use crate::backend::{Kv, Net};
use crate::model::{KvCache, Model, Sampler};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

pub struct Request {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub stop_ids: Vec<u32>,
    pub tx: Sender<Event>,
}

pub enum Event {
    Prefilled { seconds: f32, cached: usize },
    Token(u32),
    Done { finish: &'static str, batch_avg: f32, accept_rate: f32 },
}

struct Seq {
    req: Request,
    cache: Kv,              // target KV: holds committed[..len-1]; committed.last() is fed next step
    draft_cache: KvCache,   // draft KV, resynced to committed before each draft phase
    committed: Vec<u32>,    // prompt + everything emitted so far
    sampler: Sampler,
    generated: usize,
    batch_sizes: usize, // sum of batch sizes over this seq's steps, for the avg stat
    steps: usize,
    drafted: usize,
    accepted: usize,
    ngram_backoff: u32, // steps to skip n-gram drafting after a miss
    ngram_hits: usize,
    ngram_tries: usize,
}

#[derive(Clone)]
pub struct Scheduler {
    tx: Sender<Request>,
}

const MAX_BATCH: usize = 16;
const MIN_PREFIX: usize = 4;

/// KV states of recent prompts, so a new prompt sharing a prefix (system prompt, chat
/// history, few-shot examples) only has to prefill the tokens that differ.
struct PrefixCache {
    entries: Vec<(Vec<u32>, Kv, u64)>, // (tokens, kv for exactly those tokens, last use)
    clock: u64,
    budget: usize,
}

impl PrefixCache {
    fn new(default_mb: usize) -> PrefixCache {
        let mb: usize = std::env::var("PREFIX_CACHE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(default_mb);
        PrefixCache { entries: Vec::new(), clock: 0, budget: mb << 20 }
    }

    /// Longest cached prefix of `prompt` (leaving at least one token to prefill, so we get logits).
    fn lookup(&mut self, net: &Net, prompt: &[u32]) -> Option<(usize, Kv)> {
        let mut best: Option<(usize, usize)> = None; // (len, entry index)
        for (i, (toks, kv, _)) in self.entries.iter().enumerate() {
            let mut n = toks.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
            if !kv.can_truncate(n) {
                // recurrent caches are reusable only whole: entry tokens must be a prompt prefix
                if n == kv.len() { /* whole entry reused */ } else { n = 0; }
            }
            if n >= MIN_PREFIX && best.map_or(true, |(bl, _)| n > bl) {
                best = Some((n, i));
            }
        }
        let (n, i) = best?;
        self.clock += 1;
        self.entries[i].2 = self.clock;
        let mut kv = net.clone_cache(&self.entries[i].1);
        kv.truncate(n);
        Some((n, kv))
    }

    fn insert(&mut self, net: &Net, tokens: &[u32], kv: &Kv) {
        // Skip if an existing entry already covers this prompt.
        if self.entries.iter().any(|(t, _, _)| t.len() >= tokens.len() && t.starts_with(tokens)) {
            return;
        }
        // Replace entries that are prefixes of the new one.
        self.entries.retain(|(t, _, _)| !tokens.starts_with(t));
        let mut kv = net.clone_cache(kv);
        kv.truncate(tokens.len());
        self.clock += 1;
        self.entries.push((tokens.to_vec(), kv, self.clock));
        while self.entries.iter().map(|e| e.1.bytes()).sum::<usize>() > self.budget && self.entries.len() > 1 {
            let oldest = self.entries.iter().enumerate().min_by_key(|(_, e)| e.2).map(|(i, _)| i).unwrap();
            self.entries.remove(oldest);
        }
    }
}

impl Scheduler {
    pub fn start(model: Arc<Net>, draft: Option<Arc<Model>>) -> Scheduler {
        let (tx, rx) = channel::<Request>();
        std::thread::Builder::new().name("llmfast-sched".into()).spawn(move || run(model, draft, rx)).unwrap();
        Scheduler { tx }
    }

    pub fn submit(&self, req: Request) {
        let _ = self.tx.send(req);
    }
}

fn run(model: Arc<Net>, draft: Option<Arc<Model>>, rx: Receiver<Request>) {
    crate::pool::set_ftz_daz();
    let rollback = model.supports_rollback();
    if !rollback {
        eprintln!("recurrent model: speculative decoding and partial prefix reuse disabled");
    }
    let spec_k: usize = std::env::var("SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let spec_k = if draft.is_some() && rollback { spec_k } else { 0 };
    // Prompt-lookup / n-gram speculation: draft from earlier occurrences in the sequence itself.
    let ngram_n: usize = std::env::var("NGRAM_N").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let ngram_k: usize = std::env::var("NGRAM_K").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let ngram_on = rollback && std::env::var("NGRAM").map_or(true, |v| v != "0");
    let mut active: Vec<Seq> = Vec::new();
    let mut prefix = PrefixCache::new(if model.device() == "gpu" { 512 } else { 1024 });
    loop {
        // ---- admit new requests (block only when idle) ----
        let mut incoming = Vec::new();
        if active.is_empty() {
            match rx.recv() {
                Ok(r) => incoming.push(r),
                Err(_) => return,
            }
        }
        while active.len() + incoming.len() < MAX_BATCH {
            match rx.try_recv() {
                Ok(r) => incoming.push(r),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        for req in incoming {
            let t0 = Instant::now();
            let (hit, mut cache) = match prefix.lookup(&model, &req.prompt_ids) {
                Some((n, kv)) => (n, kv),
                None => (0, model.new_cache()),
            };
            let mut logits = model.forward_batch(&req.prompt_ids[hit..], &mut cache);
            prefix.insert(&model, &req.prompt_ids, &cache);
            let _ = req.tx.send(Event::Prefilled { seconds: t0.elapsed().as_secs_f32(), cached: hit });
            let mut sampler = Sampler::new(req.temperature, req.top_p, req.seed);
            // Emit the first token right away (it's free: prefill already produced its logits).
            let first = sampler.sample(&mut logits);
            let mut committed = req.prompt_ids.clone();
            let draft_cache = match &draft { Some(d) => KvCache::new(&d.config), None => KvCache::new(model.config()) };
            let mut seq = Seq { req, cache, draft_cache, committed, sampler, generated: 0, batch_sizes: 0, steps: 0, drafted: 0, accepted: 0, ngram_backoff: 0, ngram_hits: 0, ngram_tries: 0 };
            if emit(&mut seq, first).is_some() {
                finish_seq(seq, "stop");
                continue;
            }
            seq.committed.push(first);
            active.push(seq);
        }
        if active.is_empty() {
            continue;
        }

        // ---- draft phase 1: n-gram lookup in the sequence's own tokens (free) ----
        let mut drafts: Vec<Vec<u32>> = vec![Vec::new(); active.len()];
        let mut from_ngram = vec![false; active.len()];
        if ngram_on {
            for (i, s) in active.iter_mut().enumerate() {
                if s.ngram_backoff > 0 {
                    s.ngram_backoff -= 1;
                    continue;
                }
                if let Some(g) = ngram_draft(&s.committed, ngram_n, ngram_k) {
                    drafts[i] = g;
                    from_ngram[i] = true;
                    s.ngram_tries += 1;
                }
            }
        }
        // ---- draft phase 2: draft model for the rest ----
        let need_model_draft = active.iter().enumerate().any(|(i, _)| !from_ngram[i]);
        if let (Some(d), true) = (&draft, need_model_draft) {
            // resync draft caches to committed[..len-1], then feed the tail (usually 1-2 tokens)
            let mut items = Vec::new();
            for (i, s) in active.iter_mut().enumerate() {
                let need = s.committed.len() - 1;
                if s.draft_cache.len > need {
                    s.draft_cache.truncate(need);
                }
                for &t in &s.committed[s.draft_cache.len..] {
                    items.push((t, i));
                }
            }
            let mut dc: Vec<&mut KvCache> = active.iter_mut().map(|s| &mut s.draft_cache).collect();
            let mut dl = d.forward_multi(&items, &mut dc);
            for round in 0..spec_k {
                let mut next_items = Vec::new();
                for (i, l) in dl.iter().enumerate() {
                    let t = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
                    if !from_ngram[i] {
                        drafts[i].push(t);
                    }
                    next_items.push((t, i));
                }
                if round + 1 == spec_k {
                    break;
                }
                let mut dc: Vec<&mut KvCache> = active.iter_mut().map(|s| &mut s.draft_cache).collect();
                dl = d.forward_multi(&next_items, &mut dc);
            }
        }

        // ---- target phase: one batched pass over [last, d1..dK] for every sequence ----
        let mut items: Vec<(u32, usize)> = Vec::new();
        for (i, s) in active.iter().enumerate() {
            items.push((*s.committed.last().unwrap(), i));
            for &t in &drafts[i] {
                items.push((t, i));
            }
        }
        let batch = active.len();
        let mut caches: Vec<&mut Kv> = active.iter_mut().map(|s| &mut s.cache).collect();
        let any_draft = drafts.iter().any(|d| !d.is_empty());
        let all = model.forward_multi(&items, &mut caches, any_draft);

        // ---- accept/reject per sequence ----
        let mut li = 0;
        let mut keep: Vec<Seq> = Vec::new();
        for (i, mut s) in active.drain(..).enumerate() {
            let k = drafts[i].len();
            let logits = &all[li..li + k + 1];
            li += k + 1;
            s.batch_sizes += batch;
            s.steps += 1;
            s.drafted += k;
            let base = s.committed.len();
            let mut finished: Option<&'static str> = None;
            let mut accepted = 0;
            for j in 0..=k {
                let mut l = logits[j].clone();
                let t = s.sampler.sample(&mut l);
                if let Some(f) = emit(&mut s, t) {
                    finished = Some(f);
                    s.committed.push(t);
                    break;
                }
                s.committed.push(t);
                if j < k && t == drafts[i][j] {
                    accepted += 1;
                    continue;
                }
                break;
            }
            s.accepted += accepted;
            if from_ngram[i] {
                if accepted == 0 {
                    s.ngram_backoff = 3; // this region of text isn't repeating; don't pay for verification for a while
                } else {
                    s.ngram_hits += 1;
                }
            }
            // target cache holds base-1 + (k+1) positions; keep exactly committed[..len-1]
            if s.cache.can_truncate(s.committed.len() - 1) {
                s.cache.truncate(s.committed.len() - 1);
            } else {
                debug_assert_eq!(s.cache.len(), s.committed.len() - 1, "recurrent cache out of sync");
            }
            let _ = base;
            match finished {
                Some(f) => finish_seq(s, f),
                None => keep.push(s),
            }
        }
        active = keep;
    }
}

/// Send one token to the client; returns Some(finish reason) if the sequence is over.
fn emit(s: &mut Seq, t: u32) -> Option<&'static str> {
    if s.req.stop_ids.contains(&t) {
        return Some("stop");
    }
    if s.req.tx.send(Event::Token(t)).is_err() {
        return Some("disconnected");
    }
    s.generated += 1;
    if s.generated >= s.req.max_tokens {
        return Some("length");
    }
    None
}

/// If the last `n` tokens occurred earlier in `toks`, return up to `k` tokens that followed that
/// occurrence (most recent match wins). This is "prompt lookup decoding".
fn ngram_draft(toks: &[u32], n: usize, k: usize) -> Option<Vec<u32>> {
    let len = toks.len();
    for n in (2..=n).rev() {
        if len < n + 1 {
            continue;
        }
        let tail = &toks[len - n..];
        let mut i = len - n; // candidate start positions, scanning backwards
        while i > 0 {
            i -= 1;
            if &toks[i..i + n] == tail {
                let from = i + n;
                let to = (from + k).min(len - n); // don't include the tail itself
                if to > from {
                    return Some(toks[from..to].to_vec());
                }
                break;
            }
        }
    }
    None
}

fn finish_seq(s: Seq, finish: &'static str) {
    let avg = if s.steps > 0 { s.batch_sizes as f32 / s.steps as f32 } else { 0.0 };
    let rate = if s.drafted > 0 { s.accepted as f32 / s.drafted as f32 } else { 0.0 };
    let _ = s.req.tx.send(Event::Done { finish, batch_avg: avg, accept_rate: rate });
}
