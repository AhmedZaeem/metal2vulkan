//! The host's request for the GPU's deferred writes (`Device::write_back_deferred`,
//! `dirty_pages/deferred.rs`).

use super::{Device, Error};
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The most rounds of flushes one request makes: each writes back what the last found, and a
/// round finds more only where work deferred rows while the last ran.
const ROUNDS: usize = 4;

/// How long a round waits for its flush: it waits for nothing but the copies, ordered after the
/// work that wrote the rows, which completed before a host that asks after the CPU's wait
/// (a lock) asks; past it the flush completes on its own, and the request returns.
pub(crate) const WAIT: Duration = Duration::from_secs(2);

impl Device {
    /// Write back the deferred rows over `buffer`'s bytes, waited for: before its pages are
    /// rebound or detached, so the GPU's writes reach the pages they were made over.
    pub(crate) fn flush_deferred_of(&self, buffer: &crate::Buffer) {
        if !self.shared().dirty.any_deferred() {
            return;
        }
        let Some(start) = buffer.contents().map(|at| at.as_ptr() as u64) else {
            return;
        };
        // A failure leaves the rows deferred.
        let _ = self.flush_deferred(Some(start..start.saturating_add(buffer.length())));
    }

    /// Write back, unwaited, the deferred rows of homes no texture holds any more
    /// ([`DirtyShadows::orphans`](crate::dirty_pages)): their image and buffer go once it
    /// completes. Each commit on the device looks.
    pub(crate) fn flush_orphans(&self) {
        let shadows = &self.shared().dirty;
        if !shadows.any_deferred() {
            return;
        }
        let owners = shadows.orphans(|copies| !self.home_held(copies));
        if owners.is_empty() {
            return;
        }
        let flushed = (|| -> Result<(), Error> {
            let queue = self.new_command_queue()?;
            let command_buffer = queue.command_buffer()?;
            let mut blit = command_buffer.blit_command_encoder()?;
            blit.core().flush_deferred(&owners);
            blit.end_encoding()?;
            command_buffer.commit()
        })();
        if flushed.is_err() {
            shadows.unmark_orphans(&owners);
        }
    }

    /// [`Device::write_back_deferred`]: a flush of every group of deferred rows over `host`
    /// (every one for `None`), each in a command buffer of its own queue, in rounds.
    pub(crate) fn flush_deferred(&self, host: Option<Range<u64>>) -> Result<(), Error> {
        let shadows = &self.shared().dirty;
        for _ in 0..ROUNDS {
            let owners = shadows.deferred_owners(host.clone());
            if owners.is_empty() {
                break;
            }
            let queue = self.new_command_queue()?;
            let command_buffer = queue.command_buffer()?;
            let mut blit = command_buffer.blit_command_encoder()?;
            blit.core().flush_deferred(&owners);
            blit.end_encoding()?;
            command_buffer.commit()?;
            if command_buffer.wait_until_completed_for(WAIT).is_err() {
                (shadows.tally().flushes_unfinished).fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
        Ok(())
    }
}
