//! Counts of the runtime's live objects that hold device memory or the device: devices, buffers,
//! images, shadows and their texels, queues, command buffers, recorder logs and copy-in gates. A
//! run that looks for what grows reads them (`WorkCounts::live_*`); nothing decides by them.

use std::sync::atomic::{AtomicI64, Ordering};

/// What a count counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Device,
    Buffer,
    Image,
    ShadowCopies,
    Texels,
    Queue,
    CommandBuffer,
    Log,
    CopyInGate,
}

const KINDS: usize = 9;

static COUNTS: [AtomicI64; KINDS] = [const { AtomicI64::new(0) }; KINDS];

/// How many of `kind` live now, in the process.
pub(crate) fn live(kind: Kind) -> u64 {
    u64::try_from(COUNTS[kind as usize].load(Ordering::Relaxed)).unwrap_or(0)
}

/// One live object of a kind, counted from its making until it goes: a field of the object.
#[derive(Debug)]
pub(crate) struct Counted(Kind);

impl Counted {
    pub(crate) fn new(kind: Kind) -> Counted {
        COUNTS[kind as usize].fetch_add(1, Ordering::Relaxed);
        Counted(kind)
    }
}

impl Clone for Counted {
    fn clone(&self) -> Counted {
        Counted::new(self.0)
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        COUNTS[self.0 as usize].fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::{live, Counted, Kind};

    #[test]
    fn a_count_lives_with_its_object() {
        let before = live(Kind::CopyInGate);
        let counted = Counted::new(Kind::CopyInGate);
        let cloned = counted.clone();
        assert_eq!(live(Kind::CopyInGate), before + 2);
        drop((counted, cloned));
        assert_eq!(live(Kind::CopyInGate), before);
    }
}
