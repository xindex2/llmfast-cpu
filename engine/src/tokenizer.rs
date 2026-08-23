//! Byte-level BPE tokenizer (GPT-2 / Qwen style), loaded from HF tokenizer.json.
//!
//! Encoding pipeline: text → pre-tokenizer splits (words, digits, punctuation, whitespace)
//! → each chunk's bytes mapped to printable unicode chars → BPE merges by rank → vocab ids.
//! Special tokens like <|im_start|> are matched verbatim before any of that happens.

use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    specials: Vec<(String, u32)>,
    pub im_start: u32,
    pub im_end: u32,
    pub eos: u32,
}

impl Tokenizer {
    pub fn load(path: &str) -> Tokenizer {
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(path).expect("tokenizer.json")).unwrap();
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (tok, id) in j["model"]["vocab"].as_object().unwrap() {
            vocab.insert(tok.clone(), id.as_u64().unwrap() as u32);
        }
        let mut specials = Vec::new();
        for t in j["added_tokens"].as_array().unwrap() {
            let (s, id) = (t["content"].as_str().unwrap().to_string(), t["id"].as_u64().unwrap() as u32);
            vocab.insert(s.clone(), id);
            specials.push((s, id));
        }
        let mut id_to_token = vec![String::new(); vocab.values().max().map(|m| *m as usize + 1).unwrap_or(0)];
        for (t, id) in &vocab {
            id_to_token[*id as usize] = t.clone();
        }
        let mut merges = HashMap::new();
        for (rank, m) in j["model"]["merges"].as_array().unwrap().iter().enumerate() {
            let (a, b) = match m {
                serde_json::Value::String(s) => {
                    let (a, b) = s.split_once(' ').unwrap();
                    (a.to_string(), b.to_string())
                }
                serde_json::Value::Array(p) => (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()),
                _ => panic!("bad merge"),
            };
            merges.insert((a, b), rank);
        }
        let (byte_to_char, char_to_byte) = bytes_to_unicode();
        let find = |s: &str| *vocab.get(s).unwrap_or_else(|| panic!("tokenizer missing {s}"));
        Tokenizer {
            im_start: find("<|im_start|>"),
            im_end: find("<|im_end|>"),
            eos: find("<|endoftext|>"),
            vocab,
            id_to_token,
            merges,
            byte_to_char,
            char_to_byte,
            specials,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Find the earliest special token occurrence; encode plain text before it.
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(i) = rest.find(s.as_str()) {
                    if best.map_or(true, |(bi, _, _)| i < bi) {
                        best = Some((i, s.len(), *id));
                    }
                }
            }
            match best {
                Some((i, len, id)) => {
                    self.encode_plain(&rest[..i], &mut ids);
                    ids.push(id);
                    rest = &rest[i + len..];
                }
                None => {
                    self.encode_plain(rest, &mut ids);
                    break;
                }
            }
        }
        ids
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for chunk in pretokenize(text) {
            let mapped: String = chunk.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            if let Some(id) = self.vocab.get(&mapped) {
                out.push(*id);
                continue;
            }
            let mut parts: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
            loop {
                let mut best: Option<(usize, usize)> = None; // (rank, index)
                for i in 0..parts.len().saturating_sub(1) {
                    if let Some(r) = self.merges.get(&(parts[i].clone(), parts[i + 1].clone())) {
                        if best.map_or(true, |(br, _)| *r < br) {
                            best = Some((*r, i));
                        }
                    }
                }
                match best {
                    Some((_, i)) => {
                        let merged = format!("{}{}", parts[i], parts[i + 1]);
                        parts[i] = merged;
                        parts.remove(i + 1);
                    }
                    None => break,
                }
            }
            for p in parts {
                out.push(*self.vocab.get(&p).unwrap_or(&self.eos));
            }
        }
    }

    /// Raw bytes for a token id. Callers accumulate bytes and decode UTF-8 when it becomes valid,
    /// because one multibyte character can span several tokens.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let tok = &self.id_to_token[id as usize];
        if self.specials.iter().any(|(s, _)| s == tok) {
            return tok.as_bytes().to_vec();
        }
        tok.chars().map(|c| *self.char_to_byte.get(&c).unwrap_or(&b'?')).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's reversible byte ↔ printable-unicode mapping.
fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::new();
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable { char::from_u32(b).unwrap() } else { n += 1; char::from_u32(256 + n - 1).unwrap() };
        b2c[b as usize] = c;
        c2b.insert(c, b as u8);
    }
    (b2c, c2b)
}

/// Hand-written matcher for the Qwen/GPT-4 pre-tokenization regex:
///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let at = |i: usize| chars.get(i).map(|c| c.1);
    let is_nl = |c: char| c == '\r' || c == '\n';
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let start = chars[i].0;
        let mut j = i;
        let c = chars[i].1;

        // 1. contractions
        if c == '\'' {
            let lower: String = chars[i + 1..(i + 3).min(n)].iter().map(|c| c.1.to_ascii_lowercase()).collect();
            let len = if lower.starts_with("re") || lower.starts_with("ve") || lower.starts_with("ll") { 2 }
                else if lower.starts_with('s') || lower.starts_with('t') || lower.starts_with('m') || lower.starts_with('d') { 1 } else { 0 };
            if len > 0 {
                j = i + 1 + len;
                out.push(slice(text, &chars, start, j));
                i = j;
                continue;
            }
        }
        // 2. optional non-letter prefix + letters
        {
            let mut k = i;
            if !c.is_alphabetic() && !c.is_numeric() && !is_nl(c) {
                k += 1;
            }
            if at(k).map_or(false, |c| c.is_alphabetic()) {
                while at(k).map_or(false, |c| c.is_alphabetic()) {
                    k += 1;
                }
                out.push(slice(text, &chars, start, k));
                i = k;
                continue;
            }
        }
        // 3. single digit
        if c.is_numeric() {
            out.push(slice(text, &chars, start, i + 1));
            i += 1;
            continue;
        }
        // 4. optional space + punctuation run + newlines
        {
            let mut k = i;
            if c == ' ' {
                k += 1;
            }
            let is_punct = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
            if at(k).map_or(false, is_punct) {
                while at(k).map_or(false, is_punct) {
                    k += 1;
                }
                while at(k).map_or(false, is_nl) {
                    k += 1;
                }
                out.push(slice(text, &chars, start, k));
                i = k;
                continue;
            }
        }
        // 5–7. whitespace runs
        if c.is_whitespace() {
            while j < n && chars[j].1.is_whitespace() {
                j += 1;
            }
            // \s*[\r\n]+ : if the run contains a newline, take through the last newline
            let last_nl = (i..j).rev().find(|&k| is_nl(chars[k].1));
            let end = match last_nl {
                Some(k) => k + 1,
                // \s+(?!\S): leave the final space attached to the following word
                None if j < n && j - i > 1 => j - 1,
                None => j,
            };
            out.push(slice(text, &chars, start, end));
            i = end;
            continue;
        }
        // fallback: single char
        out.push(slice(text, &chars, start, i + 1));
        i += 1;
    }
    out
}

fn slice<'a>(text: &'a str, chars: &[(usize, char)], start: usize, end_idx: usize) -> &'a str {
    let end = chars.get(end_idx).map(|c| c.0).unwrap_or(text.len());
    &text[start..end]
}
