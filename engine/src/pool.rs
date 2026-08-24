//! Persistent worker pool. Spawning threads per matvec cost ~200 spawns per token; here the
//! workers live forever and pick up chunks of each job from a shared atomic counter, so
//! dispatch costs one condvar broadcast and load balances automatically.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type JobFn<'a> = dyn Fn(usize) + Sync + 'a;

struct Job {
    // Lifetime erased: `run` blocks until every chunk finished, so the closure outlives all uses.
    f: *const JobFn<'static>,
    chunks: usize,
    // static: worker w handles chunks w, w+W, w+2W... (deterministic thread↔rows mapping, so
    // first-touch NUMA placement and cache locality hold across calls). Otherwise work-stealing.
    static_: bool,
}
unsafe impl Send for Job {}

struct State {
    job: Option<Job>,
    generation: u64,
    pending: usize,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
    next_chunk: AtomicUsize,
    // static mode: per-owner cursors; owner w walks chunks w, w+W, ... then steals from others
    next_owned: Vec<AtomicUsize>,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: usize,
    // One job at a time: concurrent requests serialize here until M3 batches them properly.
    job_lock: Mutex<()>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

pub fn global() -> &'static Pool {
    POOL.get_or_init(|| {
        set_ftz_daz();
        let n = std::env::var("THREADS").ok().and_then(|v| v.parse().ok())
            .unwrap_or_else(physical_cores);
        Pool::new(n)
    })
}

/// Physical cores, not hardware threads. Decode is memory-bandwidth-bound, so the second
/// hyperthread on a core adds a barrier participant per layer without adding bandwidth --
/// on a 40-thread dual-socket box that sync cost dominates for small models. Counts unique
/// `thread_siblings_list` entries in sysfs; falls back to available_parallelism() elsewhere.
fn physical_cores() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut sibs = std::collections::HashSet::new();
    if let Ok(dir) = std::fs::read_dir("/sys/devices/system/cpu") {
        for e in dir.flatten() {
            let p = e.path().join("topology/thread_siblings_list");
            if let Ok(v) = std::fs::read_to_string(&p) {
                sibs.insert(v.trim().to_string());
            }
        }
    }
    if sibs.is_empty() { logical } else { sibs.len().min(logical).max(1) }
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { job: None, generation: 0, pending: 0 }),
            wake: Condvar::new(),
            done: Condvar::new(),
            next_chunk: AtomicUsize::new(0),
            next_owned: (0..threads).map(|_| AtomicUsize::new(0)).collect(),
        });
        // Main thread is worker 0; spawn the rest.
        pin_to_cpu(0);
        for w in 1..threads {
            let s = shared.clone();
            std::thread::spawn(move || {
                pin_to_cpu(w);
                worker(s, w, threads)
            });
        }
        Pool { shared, workers: threads, job_lock: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.workers
    }

    /// Run `f(chunk)` for chunk in 0..chunks across all workers (work-stealing); returns when done.
    pub fn run<'a>(&self, chunks: usize, f: &JobFn<'a>) {
        self.run_impl(chunks, f, false)
    }

    /// Same, but chunk→thread assignment is fixed (chunk c always runs on worker c % W).
    /// Use for weight streaming so each socket reads the rows it quantized (NUMA first-touch).
    pub fn run_static<'a>(&self, chunks: usize, f: &JobFn<'a>) {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        // STATIC=0 → plain work-stealing everywhere (best on noisy single-socket machines);
        // default on → owner-first (best on multi-socket NUMA servers).
        let on = *ON.get_or_init(|| std::env::var("STATIC").map_or(cfg!(target_os = "linux"), |v| v != "0"));
        self.run_impl(chunks, f, on)
    }

    fn run_impl<'a>(&self, chunks: usize, f: &JobFn<'a>, static_: bool) {
        if chunks == 0 {
            return;
        }
        if chunks == 1 || self.workers == 1 {
            for c in 0..chunks {
                f(c);
            }
            return;
        }
        let _guard = self.job_lock.lock().unwrap();
        let f_static: *const JobFn<'static> = unsafe { std::mem::transmute(f as *const JobFn<'a>) };
        {
            let mut st = self.shared.state.lock().unwrap();
            self.shared.next_chunk.store(0, Ordering::SeqCst);
            for (w, n) in self.shared.next_owned.iter().enumerate() {
                n.store(w, Ordering::SeqCst);
            }
            st.job = Some(Job { f: f_static, chunks, static_ });
            st.generation += 1;
            st.pending = self.workers - 1;
            self.shared.wake.notify_all();
        }
        // Participate as worker 0.
        if static_ {
            run_owned(&self.shared, 0, self.workers, chunks, f);
        } else {
            loop {
                let c = self.shared.next_chunk.fetch_add(1, Ordering::SeqCst);
                if c >= chunks {
                    break;
                }
                f(c);
            }
        }
        let mut st = self.shared.state.lock().unwrap();
        while st.pending > 0 {
            st = self.shared.done.wait(st).unwrap();
        }
        st.job = None;
    }
}

// `workers` must come from the pool: after pinning, available_parallelism() would report 1 on
// Linux, and a wrong stride makes owner-first scheduling skip chunks (garbage output).
fn worker(s: Arc<Shared>, id: usize, workers: usize) {
    set_ftz_daz();
    let mut seen = 0u64;
    loop {
        let (f, chunks, static_) = {
            let mut st = s.state.lock().unwrap();
            while st.generation == seen {
                st = s.wake.wait(st).unwrap();
            }
            seen = st.generation;
            let j = st.job.as_ref().unwrap();
            (j.f, j.chunks, j.static_)
        };
        if static_ {
            run_owned(&s, id, workers, chunks, unsafe { &*f });
        } else {
            loop {
                let c = s.next_chunk.fetch_add(1, Ordering::SeqCst);
                if c >= chunks {
                    break;
                }
                unsafe { (*f)(c) };
            }
        }
        let mut st = s.state.lock().unwrap();
        st.pending -= 1;
        if st.pending == 0 {
            s.done.notify_one();
        }
    }
}

/// Flush-to-zero + denormals-are-zero: denormal floats run ~100x slower on x86; inference
/// never needs them. Every thread that does math must set this (MXCSR is per-thread).
pub fn set_ftz_daz() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        _mm_setcsr(_mm_getcsr() | 0x8040);
    }
}

/// Owner-first scheduling: drain my own chunk stride (w, w+W, ...) then steal from other owners'
/// strides. Keeps the thread↔rows mapping stable when everyone is healthy (NUMA/cache locality)
/// but never lets one descheduled thread stall the whole step.
fn run_owned(s: &Shared, me: usize, workers: usize, chunks: usize, f: &JobFn<'_>) {
    for k in 0..workers {
        let owner = (me + k) % workers;
        if owner >= s.next_owned.len() {
            continue;
        }
        loop {
            let c = s.next_owned[owner].fetch_add(workers, Ordering::SeqCst);
            if c >= chunks {
                break;
            }
            f(c);
        }
    }
}

/// Pin the calling thread to one CPU (Linux only; PIN=0 disables). With static partitioning this
/// keeps each thread's share of the weights in its own NUMA node's memory on multi-socket servers.
#[allow(unused_variables)]
pub fn pin_to_cpu(cpu: usize) {
    #[cfg(target_os = "linux")]
    {
        if std::env::var("PIN").map_or(false, |v| v == "0") {
            return;
        }
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            let ncpu = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize;
            libc::CPU_SET(cpu % ncpu.max(1), &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn owner_first_covers_every_chunk_once() {
        let pool = Pool::new(6);
        for &chunks in &[1usize, 5, 6, 7, 64, 1000] {
            let hits: Vec<AtomicU32> = (0..chunks).map(|_| AtomicU32::new(0)).collect();
            pool.run_impl(chunks, &|c| { hits[c].fetch_add(1, Ordering::SeqCst); }, true);
            for (c, h) in hits.iter().enumerate() {
                assert_eq!(h.load(Ordering::SeqCst), 1, "chunk {c} of {chunks} hit {} times", h.load(Ordering::SeqCst));
            }
            let hits: Vec<AtomicU32> = (0..chunks).map(|_| AtomicU32::new(0)).collect();
            pool.run_impl(chunks, &|c| { hits[c].fetch_add(1, Ordering::SeqCst); }, false);
            assert!(hits.iter().all(|h| h.load(Ordering::SeqCst) == 1));
        }
    }
}
