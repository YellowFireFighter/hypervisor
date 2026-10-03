//! Where the splash's parts sit on a screen of a given size, and what colour
//! each of their pixels is at a given moment.
//!
//! Decided over a byte slice laid out like the frame buffer rather than over
//! the frame buffer itself, so every decision runs under test on the host.
//!
//! The mark is drawn from the 8×8 font scaled up by a whole number, which keeps
//! its strokes crisp at any size; its colour runs from a pale yellow at the top
//! to a deep one at the bottom, and the shine is a band leaning like the stroke
//! of a slash that sweeps across it and then rests before the next pass. The
//! bar and the spinner share one row beneath it, the bar from the mark's left
//! edge and the spinner flush with its right.

use core::time::Duration;

use font8x8::legacy::BASIC_LEGACY;
use handoff::{Channels, Framebuffer};

use crate::Progress;

/// What the mark spells.
const MARK: &[u8] = b"yxllow.dev";

/// Font pixels across and down one glyph.
const GLYPH: usize = 8;

/// Share of the screen's width the mark spans, in percent.
const MARK_WIDTH_PERCENT: usize = 45;

/// The mark is never taller than this fraction's reciprocal of the screen, so
/// a narrow, tall screen still has room for the row beneath it.
const MARK_HEIGHT_SHARE: usize = 4;

/// Space between the mark and the row beneath it, in mark scale units.
const ROW_GAP: usize = 4;

/// The bar's height, in mark scale units.
const BAR_HEIGHT: usize = 2;

/// Space between the bar's end and the spinner, in mark scale units.
const SPINNER_GAP: usize = 2;

/// How much smaller the spinner's glyph is drawn than the mark's.
const SPINNER_SHRINK: usize = 2;

/// The mark's colour along its top edge.
const MARK_TOP: [u8; 3] = [0xFF, 0xE0, 0x40];

/// The mark's colour along its bottom edge.
const MARK_BOTTOM: [u8; 3] = [0xFF, 0xB0, 0x00];

/// The colour at the heart of the shine.
const SHINE: [u8; 3] = [0xFF; 3];

/// The bar's filled part.
const FILL: [u8; 3] = [0xFF, 0xC8, 0x00];

/// The bar's unfilled part.
const TRACK: [u8; 3] = [0x30; 3];

/// The spinner's glyph.
const SPINNER_INK: [u8; 3] = [0xFF; 3];

/// The background everything is drawn over.
const BLACK: [u8; 3] = [0x00; 3];

/// How long the shine takes to cross the mark.
const SWEEP: Duration = Duration::from_millis(1200);

/// How long from one pass of the shine to the next, the rest after a sweep
/// included.
const SHINE_PERIOD: Duration = Duration::from_millis(2000);

/// The spinner's frames, in the order they turn through: `|`, `/`, `-` and
/// `\`, as bitmaps laid out like the font's.
///
/// Drawn as strokes of their own rather than taken from the font, whose `|` is
/// broken in the middle and whose four strokes differ in weight: these are two
/// pixels thick each, all through the cell's centre.
const SPINNER_FRAMES: [[u8; GLYPH]; 4] = [
    [0x18; GLYPH],
    [0x80, 0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03],
    [0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00],
    [0x01, 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0],
];

/// How long the spinner shows each frame.
const SPINNER_STEP: Duration = Duration::from_millis(100);

/// The whole of a blend, in the units [`blend`] takes its weight in.
const OPAQUE: usize = 256;

/// Bytes of one pixel in every carried format.
///
/// Widening: the constant is `u32`, and `usize` carries no lossless conversion
/// from it on this target.
const BYTES_PER_PIXEL: usize = Framebuffer::BYTES_PER_PIXEL as usize;

/// A rectangle of pixels, from its top-left corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    /// Pixels from the screen's left edge.
    x: usize,
    /// Pixels from the screen's top edge.
    y: usize,
    /// Pixels across.
    width: usize,
    /// Pixels down.
    height: usize,
}

/// One screen's splash: its geometry and where every part of it sits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scene {
    /// Visible pixels per scan line.
    width: usize,
    /// Visible scan lines.
    height: usize,
    /// Bytes from one scan line's first pixel to the next.
    pitch: usize,
    /// How a pixel's channels are laid out in those bytes.
    channels: Channels,
    /// Screen pixels across and down each font pixel of the mark.
    scale: usize,
    /// Screen pixels across and down each font pixel of the spinner.
    spinner_scale: usize,
    /// Where the mark is drawn.
    mark: Rect,
    /// Where the bar is drawn.
    bar: Rect,
    /// Where the spinner is drawn.
    spinner: Rect,
}

impl Scene {
    /// Lays the splash out on the screen `screen` describes, or `None` if the
    /// description is unusable or the screen too small for the mark.
    pub(crate) fn of(screen: &Framebuffer) -> Option<Self> {
        if !screen.usable() {
            return None;
        }
        let channels = Channels::of(screen.format)?;
        let width = usize::try_from(screen.width).ok()?;
        let height = usize::try_from(screen.height).ok()?;
        let pitch = usize::try_from(screen.pitch).ok()?;

        let columns = MARK.len() * GLYPH;
        let scale =
            (width * MARK_WIDTH_PERCENT / 100 / columns).min(height / MARK_HEIGHT_SHARE / GLYPH);
        if scale == 0 {
            return None;
        }
        let spinner_scale = (scale / SPINNER_SHRINK).max(1);

        let mark_width = columns * scale;
        let mark_height = GLYPH * scale;
        let spinner_side = GLYPH * spinner_scale;
        let bar_height = BAR_HEIGHT * scale;
        let row_height = bar_height.max(spinner_side);
        let gap = ROW_GAP * scale;
        let top = height.saturating_sub(mark_height + gap + row_height) / 2;
        let row = top + mark_height + gap;

        let mark = Rect {
            x: (width - mark_width) / 2,
            y: top,
            width: mark_width,
            height: mark_height,
        };
        let spinner = Rect {
            x: mark.x + mark_width - spinner_side,
            y: row + (row_height - spinner_side) / 2,
            width: spinner_side,
            height: spinner_side,
        };
        let bar = Rect {
            x: mark.x,
            y: row + (row_height - bar_height) / 2,
            width: mark_width.saturating_sub(spinner_side + SPINNER_GAP * scale),
            height: bar_height,
        };
        Some(Self {
            width,
            height,
            pitch,
            channels,
            scale,
            spinner_scale,
            mark,
            bar,
            spinner,
        })
    }

    /// Bytes the frame buffer spans, which is what a mapping must cover.
    pub(crate) fn span(&self) -> usize {
        self.pitch * self.height
    }

    /// Paints every byte of the frame buffer black.
    pub(crate) fn clear(&self, pixels: &mut [u8]) {
        pixels[..self.span()].fill(0);
    }

    /// Draws the frame for `elapsed`, with the bar filled to `progress`.
    pub(crate) fn draw(&self, pixels: &mut [u8], elapsed: Duration, progress: Progress) {
        self.draw_mark(pixels, elapsed);
        self.draw_bar(pixels, progress);
        self.draw_spinner(pixels, elapsed);
    }

    /// Paints every part black.
    pub(crate) fn erase(&self, pixels: &mut [u8]) {
        for part in [self.mark, self.bar, self.spinner] {
            self.fill(pixels, part, BLACK);
        }
    }

    /// Draws the mark's inked pixels, leaving the background between its
    /// strokes as it was.
    fn draw_mark(&self, pixels: &mut [u8], elapsed: Duration) {
        let shine = self.shine(elapsed);
        let glyph_width = GLYPH * self.scale;
        for (index, &character) in MARK.iter().enumerate() {
            for (row, bits) in BASIC_LEGACY[usize::from(character)].into_iter().enumerate() {
                for column in (0..GLYPH).filter(|column| bits & (1 << column) != 0) {
                    // The font spells rows top-down with bit zero leftmost.
                    let left = index * glyph_width + column * self.scale;
                    let top = row * self.scale;
                    for y in top..top + self.scale {
                        for x in left..left + self.scale {
                            let colour = self.mark_colour(x, y, shine);
                            self.put(pixels, self.mark.x + x, self.mark.y + y, colour);
                        }
                    }
                }
            }
        }
    }

    /// The colour of the mark at `(x, y)` within it, with the shine's centre
    /// at `shine` along the sweep if it is crossing.
    fn mark_colour(&self, x: usize, y: usize, shine: Option<usize>) -> [u8; 3] {
        let base = blend(MARK_TOP, MARK_BOTTOM, y * OPAQUE / self.mark.height);
        let Some(centre) = shine else {
            return base;
        };
        let band = self.band();
        // How far along the sweep this pixel lies: rightwards, and further the
        // lower it is, which is what leans the band like a slash. Offset by a
        // band so a centre entering from the left never goes negative.
        let along = x + (self.mark.height - y) / 2 + band;
        let distance = along.abs_diff(centre);
        if distance >= band {
            return base;
        }
        blend(base, SHINE, (band - distance) * OPAQUE / band)
    }

    /// Where the shine's centre is along the sweep at `elapsed`, or `None`
    /// while it rests between passes.
    ///
    /// The sweep runs from a band before the mark's leftmost pixel to a band
    /// past its rightmost, so the band enters and leaves whole.
    fn shine(&self, elapsed: Duration) -> Option<usize> {
        let phase = elapsed.as_millis() % SHINE_PERIOD.as_millis();
        let sweep = SWEEP.as_millis();
        if phase >= sweep {
            return None;
        }
        let travel = self.mark.width + self.mark.height / 2 + 2 * self.band();
        usize::try_from(phase * u128::try_from(travel).ok()? / sweep).ok()
    }

    /// Half the shine band's width along the sweep.
    fn band(&self) -> usize {
        self.mark.height
    }

    /// Draws the bar, filled from the left to `progress`.
    fn draw_bar(&self, pixels: &mut [u8], progress: Progress) {
        let filled = progress.of_width(self.bar.width);
        self.fill(
            pixels,
            Rect {
                width: filled,
                ..self.bar
            },
            FILL,
        );
        self.fill(
            pixels,
            Rect {
                x: self.bar.x + filled,
                width: self.bar.width - filled,
                ..self.bar
            },
            TRACK,
        );
    }

    /// Draws the spinner's frame for `elapsed`, background included, so no
    /// stroke of the frame before it is left behind.
    fn draw_spinner(&self, pixels: &mut [u8], elapsed: Duration) {
        let steps = elapsed.as_millis() / SPINNER_STEP.as_millis();
        let frames = SPINNER_FRAMES.len() as u128;
        let frame = SPINNER_FRAMES[usize::try_from(steps % frames).unwrap_or(0)];
        for (row, bits) in frame.into_iter().enumerate() {
            for column in 0..GLYPH {
                let colour = if bits & (1 << column) != 0 {
                    SPINNER_INK
                } else {
                    BLACK
                };
                self.fill(
                    pixels,
                    Rect {
                        x: self.spinner.x + column * self.spinner_scale,
                        y: self.spinner.y + row * self.spinner_scale,
                        width: self.spinner_scale,
                        height: self.spinner_scale,
                    },
                    colour,
                );
            }
        }
    }

    /// Paints every pixel of `area` one colour.
    fn fill(&self, pixels: &mut [u8], area: Rect, colour: [u8; 3]) {
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                self.put(pixels, x, y, colour);
            }
        }
    }

    /// Stores one pixel, ignoring coordinates off the visible area rather
    /// than wrapping them into a neighbouring scan line.
    fn put(&self, pixels: &mut [u8], x: usize, y: usize, colour: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let offset = y * self.pitch + x * BYTES_PER_PIXEL;
        pixels[offset..offset + BYTES_PER_PIXEL].copy_from_slice(&self.channels.encode(colour));
    }
}

/// `from` moved `weight` of the way to `to`, with [`OPAQUE`] being all of it.
fn blend(from: [u8; 3], to: [u8; 3], weight: usize) -> [u8; 3] {
    let weight = weight.min(OPAQUE);
    core::array::from_fn(|channel| {
        let mixed = (usize::from(from[channel]) * (OPAQUE - weight)
            + usize::from(to[channel]) * weight)
            / OPAQUE;
        u8::try_from(mixed).unwrap_or(u8::MAX)
    })
}

#[cfg(test)]
mod tests {
    //! Layout and drawing decisions, over buffers that stand in for the
    //! mapping.

    extern crate std;

    use core::time::Duration;
    use std::{vec, vec::Vec};

    use handoff::Framebuffer;

    use super::{
        BLACK, FILL, MARK_BOTTOM, MARK_TOP, Rect, SHINE_PERIOD, SPINNER_STEP, SWEEP, Scene, TRACK,
        blend,
    };
    use crate::Progress;

    /// A screen of the size QEMU's standard display starts in, with padding
    /// past the visible pixels of each scan line.
    fn screen() -> Framebuffer {
        Framebuffer {
            base: 0x8000_0000,
            pitch: 1280 * 4 + 64,
            width: 1280,
            height: 800,
            format: Framebuffer::BGRX,
        }
    }

    /// The scene for [`screen`], and a buffer standing in for its mapping.
    fn canvas() -> (Scene, Vec<u8>) {
        let scene = Scene::of(&screen()).expect("the screen is usable");
        let pixels = vec![0; scene.span()];
        (scene, pixels)
    }

    /// The red, green and blue of the pixel at `(x, y)`.
    fn pixel(scene: &Scene, pixels: &[u8], x: usize, y: usize) -> [u8; 3] {
        let offset = y * scene.pitch + x * 4;
        [pixels[offset + 2], pixels[offset + 1], pixels[offset]]
    }

    /// Every pixel of `area`.
    fn within(scene: &Scene, pixels: &[u8], area: Rect) -> Vec<[u8; 3]> {
        let mut seen = Vec::new();
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                seen.push(pixel(scene, pixels, x, y));
            }
        }
        seen
    }

    /// Whether `colour` is lighter than any yellow the mark has without the
    /// shine, which only the shine's white mixed in can make it.
    fn shining(colour: [u8; 3]) -> bool {
        colour[2] > MARK_TOP[2]
    }

    #[test]
    fn the_parts_are_centred_and_stacked_on_the_screen() {
        let (scene, _) = canvas();
        let Scene {
            mark,
            bar,
            spinner,
            width,
            height,
            ..
        } = scene;
        assert_eq!(mark.x * 2 + mark.width, width);
        assert!(mark.y + mark.height < bar.y);
        assert!(bar.x + bar.width < spinner.x);
        assert_eq!(spinner.x + spinner.width, mark.x + mark.width);
        assert!(spinner.y + spinner.height <= height);
        assert!(bar.width > 0);
    }

    #[test]
    fn a_screen_too_small_for_the_mark_draws_no_splash() {
        let tiny = Framebuffer {
            width: 64,
            height: 48,
            pitch: 64 * 4,
            ..screen()
        };
        assert!(Scene::of(&tiny).is_none());
        let unusable = Framebuffer {
            base: 0,
            ..screen()
        };
        assert!(Scene::of(&unusable).is_none());
    }

    #[test]
    fn at_rest_the_mark_is_yellow_from_its_top_colour_to_its_bottom_one() {
        let (scene, mut pixels) = canvas();
        scene.draw(&mut pixels, SWEEP, Progress::NONE);
        let inked: Vec<_> = within(&scene, &pixels, scene.mark)
            .into_iter()
            .filter(|&colour| colour != BLACK)
            .collect();
        // The top row of the tall letters carries the top colour exactly.
        assert!(inked.contains(&MARK_TOP));
        assert!(inked.iter().all(|&colour| {
            let [red, green, _] = colour;
            red == 0xFF && (MARK_BOTTOM[1]..=MARK_TOP[1]).contains(&green) && !shining(colour)
        }));
    }

    #[test]
    fn the_shine_crosses_the_mark_from_left_to_right() {
        let (scene, mut pixels) = canvas();
        let half = Rect {
            width: scene.mark.width / 2,
            ..scene.mark
        };
        let right = Rect {
            x: scene.mark.x + half.width,
            ..half
        };
        let lit = |pixels: &[u8], area| {
            within(&scene, pixels, area)
                .into_iter()
                .filter(|&colour| shining(colour))
                .count()
        };

        scene.draw(&mut pixels, SWEEP / 4, Progress::NONE);
        assert!(lit(&pixels, half) > lit(&pixels, right));
        scene.draw(&mut pixels, SWEEP * 3 / 4, Progress::NONE);
        assert!(lit(&pixels, right) > lit(&pixels, half));
        // And it comes round again on the next pass.
        scene.draw(&mut pixels, SHINE_PERIOD + SWEEP / 4, Progress::NONE);
        assert!(lit(&pixels, half) > lit(&pixels, right));
    }

    #[test]
    fn the_bar_fills_to_the_progress_and_tracks_the_rest() {
        let (scene, mut pixels) = canvas();
        let bar = scene.bar;
        let middle = bar.y + bar.height / 2;

        scene.draw(&mut pixels, Duration::ZERO, Progress::NONE);
        assert_eq!(pixel(&scene, &pixels, bar.x, middle), TRACK);

        scene.draw(&mut pixels, Duration::ZERO, Progress::of(1, 2));
        assert_eq!(pixel(&scene, &pixels, bar.x, middle), FILL);
        assert_eq!(
            pixel(&scene, &pixels, bar.x + bar.width / 2 - 1, middle),
            FILL
        );
        assert_eq!(
            pixel(&scene, &pixels, bar.x + bar.width / 2 + 1, middle),
            TRACK
        );

        scene.draw(&mut pixels, Duration::ZERO, Progress::COMPLETE);
        assert!(
            within(&scene, &pixels, bar)
                .into_iter()
                .all(|colour| colour == FILL)
        );
    }

    #[test]
    fn the_spinner_turns_through_four_frames_and_comes_round() {
        let (scene, mut pixels) = canvas();
        let frame = |pixels: &mut Vec<u8>, step: u32| {
            scene.draw(pixels, SPINNER_STEP * step, Progress::NONE);
            within(&scene, pixels, scene.spinner)
        };
        let frames: Vec<_> = (0..5).map(|step| frame(&mut pixels, step)).collect();
        for (index, one) in frames[..4].iter().enumerate() {
            for other in &frames[index + 1..4] {
                assert_ne!(one, other);
            }
        }
        assert_eq!(frames[0], frames[4]);
    }

    #[test]
    fn nothing_outside_the_parts_is_drawn() {
        let (scene, mut pixels) = canvas();
        scene.draw(&mut pixels, SWEEP / 2, Progress::of(1, 3));
        for y in 0..scene.height {
            for x in 0..scene.width {
                let inside = [scene.mark, scene.bar, scene.spinner].iter().any(|part| {
                    (part.x..part.x + part.width).contains(&x)
                        && (part.y..part.y + part.height).contains(&y)
                });
                if !inside {
                    assert_eq!(pixel(&scene, &pixels, x, y), BLACK, "({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn erasing_hands_back_a_black_screen() {
        let (scene, mut pixels) = canvas();
        scene.draw(&mut pixels, SWEEP / 2, Progress::COMPLETE);
        scene.erase(&mut pixels);
        assert!(pixels.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn a_blend_runs_from_one_colour_to_the_other() {
        assert_eq!(blend(MARK_TOP, [0xFF; 3], 0), MARK_TOP);
        assert_eq!(blend(MARK_TOP, [0xFF; 3], 256), [0xFF; 3]);
        assert_eq!(blend([0; 3], [200; 3], 128), [100; 3]);
    }
}
