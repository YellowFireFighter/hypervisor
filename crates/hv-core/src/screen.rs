//! What the machine's screen shows while the host comes up.
//!
//! One of two things, chosen at build time: the boot screen with the `splash`
//! feature, or the log with `efifb`, which draws its records onto the same
//! pixels and so cannot share them. With neither, the screen keeps whatever
//! the loader and firmware left on it.
//!
//! The boot screen is animated here rather than in the loader because this is
//! where a clock exists. A frame is drawn at every step of bring-up the caller
//! reports, and then, before the guest is entered, frame after frame until the
//! bar has filled over [`MINIMUM`] — so a machine that comes up in a fraction
//! of a second still shows the whole of it. The bar never runs ahead of either
//! the steps or that pace, which is what makes it fill smoothly on a fast
//! machine and honestly on a slow one.

use core::time::Duration;

use clock::Instant;
use handoff::Framebuffer;
use log::{info, warn};
use paging::{AddressSpace, CacheType, Protection};
use splash::{Progress, Splash};
use x86_64::PhysAddr;

#[cfg(all(feature = "splash", feature = "efifb"))]
compile_error!("`splash` and `efifb` both draw on the screen; build with one of them");

/// The least time the boot screen stays up, from the clock's start to the
/// guest's.
const MINIMUM: Duration = Duration::from_secs(3);

/// Microseconds between two frames of the boot screen while it is held.
const FRAME_MICROS: u64 = 16_000;

/// The steps of bring-up the boot screen's bar counts, in the order they
/// complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// The host's descriptor tables are live.
    Tables,
    /// The machine is surveyed and the clock started.
    Clock,
    /// This processor's interrupt controller, real and emulated, is up.
    Interrupts,
    /// The guest's memory is described.
    Partition,
    /// Virtualization is on and the guest's processor seeded.
    Processor,
    /// The machine's devices are enumerated.
    Devices,
    /// uACPI has read the machine's definition blocks.
    Namespace,
    /// The host checked itself and is about to enter the guest.
    Ready,
}

impl Step {
    /// Steps in the whole of bring-up.
    const COUNT: u64 = Self::Ready as u64 + 1;

    /// How far through bring-up this step's completion is.
    fn progress(self) -> Progress {
        Progress::of(self as u64 + 1, Self::COUNT)
    }
}

/// The screen while the host comes up.
pub(crate) struct Screen {
    /// The boot screen, where one is built and the screen could take it.
    splash: Option<Splash>,
    /// How far bring-up has come.
    reached: Progress,
    /// When the boot screen's first frame with a clock behind it was drawn,
    /// which is where its animation and its hold are timed from.
    started: Option<Instant>,
}

impl Screen {
    /// Maps the handoff's frame buffer and gives it to whichever of the log
    /// and the boot screen this image is built with.
    ///
    /// The aperture is device memory, which is why this mapping exists at all:
    /// the direct map covers RAM only. It is mapped `UncachedMinus`, which
    /// lets firmware's MTRRs keep the type they chose for this range —
    /// write-combining where firmware wanted scanout performance, uncached
    /// where it did not — and the translation stands for as long as the
    /// machine does, because neither the log's writer nor the boot screen
    /// holds a reference to it that could go stale.
    ///
    /// A machine with no usable screen in its handoff, or one whose aperture
    /// will not map, logs through whatever port answered and shows no boot
    /// screen; neither is worth a boot.
    pub(crate) fn attach(space: &mut AddressSpace, screen: &Framebuffer) -> Self {
        let mut attached = Self {
            splash: None,
            reached: Progress::NONE,
            started: None,
        };
        if !cfg!(any(feature = "efifb", feature = "splash")) {
            return attached;
        }
        if !screen.usable() {
            info!("core: no usable screen in the handoff; logging stays on the ports");
            return attached;
        }
        let span = u64::from(screen.pitch) * u64::from(screen.height);
        // SAFETY: the range is device memory firmware itself drew through as a
        // linear frame buffer, described by firmware's own console mode and
        // carried verbatim in the handoff; nothing else maps or writes this
        // aperture; read-write non-executable is what drawing needs; and the
        // returned handle is dropped without unmapping, which is deliberate —
        // the translation must outlive every log line and every frame.
        let address = match unsafe {
            space.map_physical(
                PhysAddr::new(screen.base),
                span,
                Protection::ReadWrite,
                CacheType::UncachedMinus,
            )
        } {
            Ok(mapping) => mapping.addr().as_u64(),
            Err(error) => {
                warn!("core: the frame buffer would not map: {error}");
                return attached;
            }
        };
        if cfg!(feature = "efifb") && serial::offer_screen(screen, address) {
            info!("core: logging attached to the frame buffer at {address:#x}");
        }
        if cfg!(feature = "splash") {
            // SAFETY: `address` was just mapped writable over the whole of the
            // frame buffer `screen` describes and is never unmapped; with the
            // log not built in, the boot screen is the only writer of it until
            // `hand_over` gives the display to the guest.
            attached.splash = unsafe { Splash::new(screen, address) };
            attached.frame();
        }
        attached
    }

    /// Records that bring-up has finished `step` and draws a frame showing
    /// it.
    pub(crate) fn advance(&mut self, step: Step) {
        self.reached = step.progress();
        self.frame();
    }

    /// Holds the boot screen until its bar has filled over [`MINIMUM`], then
    /// blanks it, handing the display to the guest.
    pub(crate) fn hand_over(mut self) {
        if self.splash.is_none() {
            return;
        }
        self.reached = Progress::COMPLETE;
        while self.frame() != Progress::COMPLETE {
            // A clock that is not there cannot pace a hold, so there is none.
            if clock::sleep_micros(FRAME_MICROS).is_err() {
                break;
            }
        }
        if let Some(splash) = &mut self.splash {
            splash.erase();
        }
    }

    /// Draws the boot screen's frame for now, and answers how full its bar
    /// is.
    fn frame(&mut self) -> Progress {
        let elapsed = match (clock::now(), self.started) {
            (Some(now), Some(started)) => now - started,
            (Some(now), None) => {
                self.started = Some(now);
                Duration::ZERO
            }
            (None, _) => Duration::ZERO,
        };
        let shown = self.reached.min(Progress::of_time(elapsed, MINIMUM));
        if let Some(splash) = &mut self.splash {
            splash.draw(elapsed, shown);
        }
        shown
    }
}
