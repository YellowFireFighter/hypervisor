//! The boot screen: `yxllow.dev` in yellow with a white shine sweeping across
//! it, and beneath it a loading bar with a spinner turning at its end.
//!
//! The screen is the frame buffer firmware's console left behind, described by
//! the handoff and drawn to as pixels. Whoever owns the display while the host
//! comes up builds a [`Splash`] over it, clears it once, and then draws a frame
//! whenever there is something new to show: how far the boot has come as a
//! [`Progress`], and how long it has been running as the animation's clock.
//! Nothing here keeps time or decides how long the screen stays up — a frame is
//! a pure function of those two values, so the caller paces it.
//!
//! What a frame covers is the mark, the bar and the spinner and nothing else:
//! the rest of the screen stays as [`Splash::clear`] left it, so a frame costs
//! stores in proportion to the parts rather than to the display, which is what
//! keeps it cheap on an aperture mapped uncached.

#![no_std]

mod scene;

use core::time::Duration;

use handoff::Framebuffer;

use crate::scene::Scene;

/// How far the boot has come, in thousandths of the whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Progress(u16);

impl Progress {
    /// Nothing done yet.
    pub const NONE: Self = Self(0);
    /// Everything done.
    pub const COMPLETE: Self = Self(Self::PARTS);

    /// Thousandths in the whole.
    const PARTS: u16 = 1000;

    /// `done` steps of `total`, with anything past the end counted as the end
    /// and a total of zero counted as complete.
    #[must_use]
    pub fn of(done: u64, total: u64) -> Self {
        if total == 0 {
            return Self::COMPLETE;
        }
        let parts = u128::from(done.min(total)) * u128::from(Self::PARTS) / u128::from(total);
        Self(u16::try_from(parts).unwrap_or(Self::PARTS))
    }

    /// How much of `whole` has gone by after `elapsed`.
    #[must_use]
    pub fn of_time(elapsed: Duration, whole: Duration) -> Self {
        let nanos = |span: Duration| u64::try_from(span.as_nanos()).unwrap_or(u64::MAX);
        Self::of(nanos(elapsed), nanos(whole))
    }

    /// How many of `width` pixels this much progress fills.
    pub(crate) fn of_width(self, width: usize) -> usize {
        width * usize::from(self.0) / usize::from(Self::PARTS)
    }
}

/// The boot screen, drawn onto a mapped frame buffer.
pub struct Splash {
    /// Where every part sits and what colour it is.
    scene: Scene,
    /// The first byte of the frame buffer's mapping.
    address: u64,
}

impl Splash {
    /// A splash over the frame buffer `screen` describes, written through
    /// `address`, or `None` for a screen with no usable description or too
    /// small to draw the mark on.
    ///
    /// Nothing is drawn yet.
    ///
    /// # Safety
    ///
    /// `address` must be the first byte of a writable mapping of the whole of
    /// that frame buffer — `pitch * height` bytes — which stays mapped for as
    /// long as the splash exists, and nothing else may write those bytes while
    /// it draws.
    #[must_use]
    pub unsafe fn new(screen: &Framebuffer, address: u64) -> Option<Self> {
        Some(Self {
            scene: Scene::of(screen)?,
            address,
        })
    }

    /// Paints the whole screen black, which is the background every frame is
    /// drawn over.
    pub fn clear(&mut self) {
        self.pixels(Scene::clear);
    }

    /// Draws the frame for `elapsed` into the boot, with the bar filled to
    /// `progress`.
    pub fn draw(&mut self, elapsed: Duration, progress: Progress) {
        self.pixels(|scene, pixels| scene.draw(pixels, elapsed, progress));
    }

    /// Paints every part black again, handing the display back as blank as
    /// [`Splash::clear`] made it.
    pub fn erase(&mut self) {
        self.pixels(Scene::erase);
    }

    /// Runs `paint` over the frame buffer's bytes.
    fn pixels(&mut self, paint: impl FnOnce(&Scene, &mut [u8])) {
        // SAFETY: `new`'s caller vouched that `address` is the first byte of a
        // writable mapping spanning the frame buffer, which `Scene::span` is,
        // that it stays mapped while this splash exists and that nothing else
        // writes it meanwhile; the slice does not outlive this call, and
        // `&mut self` keeps two of them from existing at once.
        let pixels =
            unsafe { core::slice::from_raw_parts_mut(self.address as *mut u8, self.scene.span()) };
        paint(&self.scene, pixels);
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use super::Progress;

    #[test]
    fn progress_counts_thousandths_of_the_steps_done() {
        assert_eq!(Progress::of(0, 4), Progress::NONE);
        assert_eq!(Progress::of(1, 2), Progress(500));
        assert_eq!(Progress::of(4, 4), Progress::COMPLETE);
    }

    #[test]
    fn progress_past_the_end_or_of_nothing_is_complete() {
        assert_eq!(Progress::of(7, 4), Progress::COMPLETE);
        assert_eq!(Progress::of(0, 0), Progress::COMPLETE);
    }

    #[test]
    fn progress_over_time_is_the_share_of_the_whole_gone_by() {
        let whole = Duration::from_secs(3);
        assert_eq!(
            Progress::of_time(Duration::from_millis(1500), whole),
            Progress(500)
        );
        assert_eq!(
            Progress::of_time(Duration::from_secs(9), whole),
            Progress::COMPLETE
        );
    }

    #[test]
    fn progress_fills_its_share_of_a_width() {
        assert_eq!(Progress(250).of_width(400), 100);
        assert_eq!(Progress::COMPLETE.of_width(400), 400);
        assert_eq!(Progress::NONE.of_width(400), 0);
    }
}
