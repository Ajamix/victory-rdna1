//! Lazy copy-back (`VICTORY_BACK_LAZY=1`): what shaders wrote to card copies
//! goes into title memory when something touches it, not at every landing
//! (as yuzu, Ryujinx and shadPS4 keep GPU-written pages on the GPU until the
//! CPU reads them). The title reads almost none of it (runs victory36/37),
//! and a landing's merge wrote ~40 MB a frame that our refresh then sent up
//! again.
//!
//! A landed piece's part over whole 64 KiB blocks is parked (its fetched
//! bytes, copied) and those blocks made no-access with the read probe's
//! machinery (`watch::lazy_close`); the bytes at either end are merged at
//! once. The parked merge runs, in landing order, before anything opens a
//! block: a touch (the handler, `fault`), a kernel call of ours
//! (`watch::probe_open_by` -> `unpark`) or a merge of a later landing
//! (`Shadow::card_merge_relay` opens its range first). A later landing over
//! the same blocks of the same copy drops the parked pieces it covers: the
//! merge writes only bytes the card changed, so merging only the newer bytes
//! leaves title memory as merging both would.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::shadow::Shadow;
use crate::watch::PAGE as BLOCK;

/// `VICTORY_BACK_LAZY`: 0 off, 1 on, 2 `alt`.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_BACK_LAZY").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        };
        if mode != 0 {
            crate::watch::install_handler();
            tracing::info!("lazy copy-back: shaders' writes to card copies go into title memory on first touch (VICTORY_BACK_LAZY={mode})");
        }
        mode
    })
}

/// Whether lazy copy-back is set up (the handler, opens before our calls):
/// parked pieces may exist.
pub(crate) fn on() -> bool {
    mode() != 0
}

/// Whether landings park now (`alt`: twenty seconds so, twenty merging at
/// landing as usual; pieces parked before still go in on a touch).
pub(crate) fn parking() -> bool {
    match mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "card pieces parked at landing from now (VICTORY_BACK_LAZY=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

struct Parked {
    seq: u64,
    shadow: Arc<Shadow>,
    /// Offset in the copy and the card's bytes there.
    at: usize,
    bytes: Vec<u8>,
    after: u64,
    /// Blocks `first..end` (address / BLOCK).
    first: usize,
    end: usize,
}

static PARKED: Mutex<BTreeMap<usize, Vec<Arc<Parked>>>> = Mutex::new(BTreeMap::new());
static SEQ: AtomicU64 = AtomicU64::new(0);
/// [parked pieces, parked MB, dropped (covered by a newer one), merged on a touch,
/// merged for a kernel call or merge of ours, merged before a newer landing, merged without the lock]
pub(crate) static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Takes every parked piece over blocks `first..end` out of the map, in landing order.
fn take(first: usize, end: usize) -> Vec<Arc<Parked>> {
    let mut map = PARKED.lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<Arc<Parked>> = Vec::new();
    for (_, v) in map.range(first..end) {
        for p in v {
            if !out.iter().any(|o| Arc::ptr_eq(o, p)) {
                out.push(p.clone());
            }
        }
    }
    for p in &out {
        for b in p.first..p.end {
            if let Some(v) = map.get_mut(&b) {
                v.retain(|q| !Arc::ptr_eq(q, p));
                if v.is_empty() {
                    map.remove(&b);
                }
            }
        }
    }
    out.sort_by_key(|p| p.seq);
    out
}

fn merge(p: &Parked, title: &mut Vec<u8>) -> usize {
    let (written, locked) = p.shadow.card_merge_relay_lazy(p.at, &p.bytes, p.after, title);
    if !locked {
        STATS[6].fetch_add(1, Ordering::Relaxed);
    }
    written
}

/// Merges the parked pieces over `at..at + len` (before it is opened).
pub(crate) fn unpark(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let pieces = take(at / BLOCK, (at + len - 1) / BLOCK + 1);
    let mut title = Vec::new();
    for p in pieces {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        merge(&p, &mut title);
    }
}

/// The handler's part: a touch of a block with parked pieces merges them
/// (their merge opens the block). `false` if none was parked there.
pub(crate) fn fault(at: usize) -> bool {
    let b = at / BLOCK;
    let pieces = take(b, b + 1);
    if pieces.is_empty() {
        return false;
    }
    let mut title = Vec::new();
    for p in pieces {
        STATS[3].fetch_add(1, Ordering::Relaxed);
        merge(&p, &mut title);
    }
    true
}

/// A landing's piece in lazy mode: [`Shadow::card_merge_relay`] for the ends,
/// parked for the whole blocks between. Returns the bytes written now.
pub(crate) fn land(shadow: &Arc<Shadow>, at: usize, gpu: &[u8], after: u64, title: &mut Vec<u8>) -> usize {
    let base = shadow.start_address();
    let (a, e) = (base + at, base + at + gpu.len());
    let (first, end) = (a.div_ceil(BLOCK), e / BLOCK);
    if first >= end {
        return shadow.card_merge_relay(at, gpu, after, title);
    }
    let (m0, m1) = (first * BLOCK - base, end * BLOCK - base);
    let mut written = 0;
    if m0 > at {
        written += shadow.card_merge_relay(at, &gpu[..m0 - at], after, title);
    }
    if at + gpu.len() > m1 {
        written += shadow.card_merge_relay(m1, &gpu[m1 - at..], after, title);
    }
    // A merge of this copy on another thread (a touch, a kernel call of
    // ours) finishes before its blocks close again; one that comes after
    // finds the new piece. (Merges below on this thread try the lock.)
    let _merging = shadow.merge_lock();
    for p in take(first, end) {
        if Arc::ptr_eq(&p.shadow, shadow) && p.first >= first && p.end <= end {
            STATS[2].fetch_add(1, Ordering::Relaxed);
        } else {
            STATS[5].fetch_add(1, Ordering::Relaxed);
            written += merge(&p, title);
        }
    }
    let mid = &gpu[m0 - at..m1 - at];
    let piece = Arc::new(Parked {
        seq: SEQ.fetch_add(1, Ordering::Relaxed),
        shadow: shadow.clone(),
        at: m0,
        bytes: mid.to_vec(),
        after,
        first,
        end,
    });
    // In the map before the blocks close: a touch after the close finds it.
    {
        let mut map = PARKED.lock().unwrap_or_else(|e| e.into_inner());
        for b in first..end {
            map.entry(b).or_default().push(piece.clone());
        }
    }
    if crate::watch::lazy_close(first * BLOCK, (end - first) * BLOCK) {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(mid.len() as u64, Ordering::Relaxed);
    } else {
        // A block that cannot be made no-access (guarded, not committed
        // read-write): merged now, as without the switch.
        for p in take(first, end) {
            written += merge(&p, title);
        }
    }
    written
}

/// With the frame line: per frame over `frames`, what lazy mode did.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    let parked_now: usize = PARKED.lock().unwrap_or_else(|e| e.into_inner()).len();
    tracing::info!(
        "lazy copy-back per frame (VICTORY_BACK_LAZY): {:.1} pieces parked ({:.1} MB), {:.1} dropped as covered by a newer landing, merged {:.2} on a touch, {:.2} for our kernel calls, {:.2} before a newer landing, {} without the copy's lock; {} blocks parked now",
        s[0] as f64 / frames,
        s[1] as f64 / 1e6 / frames,
        s[2] as f64 / frames,
        s[3] as f64 / frames,
        s[4] as f64 / frames,
        s[5] as f64 / frames,
        s[6],
        parked_now
    );
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualQuery,
    };

    fn protect(at: usize) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        info.Protect
    }

    /// Memory freed and taken again at the same address while its block was
    /// noted closed: the next close closes it again, not trusting the note.
    #[test]
    fn close_after_free_and_alloc_again() {
        const LEN: usize = 2 * super::BLOCK;
        // SAFETY: a fresh read-write reservation of ours.
        let at = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        assert!(crate::watch::lazy_close(at, LEN));
        assert_eq!(protect(at), PAGE_NOACCESS);
        // SAFETY: ours; taken again at the same address at once.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
        let again = unsafe { VirtualAlloc(at as *const _, LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_eq!(again, at, "the address was taken by someone else meanwhile");
        assert_eq!(protect(at), PAGE_READWRITE);
        assert!(crate::watch::lazy_close(at, LEN));
        assert_eq!(protect(at), PAGE_NOACCESS, "a stale note left the block open");
        // SAFETY: ours.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }
}
