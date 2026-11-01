//! Drops that may run the program's code, moved off the thread that let go of them.
//!
//! A buffer over the program's memory runs the program's code as it goes: its deallocator, its
//! keep-alive, and a consumer's range token, which may take the consumer's own locks. So does a
//! dirty log's watch (the consumer's `DirtyPages::Watch`). The runtime holds such buffers in
//! copies' bases and deferred write-backs (`dirty_pages.rs`, `texture/homes.rs`). Their last
//! handles can go where the runtime holds one of its locks, or where its caller holds one of its
//! own (a texture made inside the consumer's lock asks the homes registry, which looks at other
//! homes' buffers). There such a drop could wait on a lock its own thread holds, or one another
//! thread holds while it waits on the runtime's. So [`Later`] and [`drop_later`] hand those
//! values to the process's dropper thread, which drops them in the order they came, holding no
//! lock of the runtime's.

use super::device::lock;
use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;

/// A value dropped on the dropper thread ([`drop_later`]) when it goes.
pub(crate) struct Later<T: Send + 'static>(Option<T>);

impl<T: Send + 'static> Later<T> {
    pub(crate) fn new(value: T) -> Later<T> {
        Later(Some(value))
    }
}

impl<T: Send + 'static> Drop for Later<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop_later(value);
        }
    }
}

/// The values the dropper thread is to drop.
type Dropped = Box<dyn Send>;

/// Drop `value` on the process's dropper thread, made at the first, in order after every value
/// sent before it; here, where no thread can be made.
pub(crate) fn drop_later<T: Send + 'static>(value: T) {
    static DROPPER: Mutex<Option<Sender<Dropped>>> = Mutex::new(None);
    let mut dropper = lock(&DROPPER);
    if dropper.is_none() {
        let (sender, received) = channel::<Dropped>();
        let builder = std::thread::Builder::new().name("m2v-dropper".to_owned());
        let spawned = crate::exit_guard::spawn(builder, move || {
            for value in received {
                drop(value);
            }
        });
        if spawned.is_ok() {
            *dropper = Some(sender);
        }
    }
    let unsent = match dropper.as_ref() {
        Some(sender) => sender.send(Box::new(value)).err().map(|error| error.0),
        None => Some(Box::new(value) as Dropped),
    };
    drop(dropper);
    drop(unsent);
}

#[cfg(test)]
mod tests {
    use super::{drop_later, Later};
    use std::sync::mpsc::{channel, Sender};
    use std::thread::ThreadId;

    /// Tells where and in what order it was dropped.
    struct Told(u32, Sender<(u32, ThreadId)>);

    impl Drop for Told {
        fn drop(&mut self) {
            let _ = self.1.send((self.0, std::thread::current().id()));
        }
    }

    #[test]
    fn a_value_let_go_of_later_goes_on_another_thread_in_order() {
        let (sender, told) = channel();
        drop(Later::new(Told(1, sender.clone())));
        drop_later(Told(2, sender.clone()));
        drop(Later::new(Told(3, sender)));
        let here = std::thread::current().id();
        let order: Vec<u32> = (0..3)
            .map(|_| {
                let (n, on) = told.recv().unwrap();
                assert_ne!(on, here, "dropped off the thread that let go of it");
                n
            })
            .collect();
        assert_eq!(order, [1, 2, 3]);
    }
}
