//! The UEFI frame buffer, which is a screen and no port at all.
//!
//! Firmware's console drew its last frame into a linear frame buffer whose
//! description travels in the handoff; this backend draws log lines into the
//! same pixels, so a machine with neither a debug console nor a UART header
//! still has something to say through. [`Efifb::describe`] is built from that
//! description plus an address bytes are written through — the physical base
//! under the loader, an explicit mapping under the hypervisor image, where the
//! aperture is device memory the direct map deliberately does not cover.
//!
//! The mapping side decides what the bytes cost; this module only assumes the
//! result behaves like memory. The hypervisor maps it `UncachedMinus`: the
//! PAT says uncached while firmware's MTRRs keep their veto, so a range the
//! firmware marked write-combining for scanout stays write-combining. Either
//! way plain stores reach the screen without a flush on x86, which is why
//! nothing here is volatile: the writes are ordinary pixel updates, the
//! scanout re-reads them continuously, and no ordering between two of our own
//! stores can change what a finished glyph looks like.
//!
//! The backend owns the display only until the guest starts drawing. It is
//! attached after both ports decline, and [`serial::retire_screen`] takes it
//! back before the guest is entered — past that point every line is discarded
//! rather than fought over with whatever the guest puts on the display.
//!
//! Two writers exist. The locked one runs one whole line at a time under the
//! crate's output lock, like the ports do. The other is [`serial::emergency`],
//! which takes no lock, reconstructs its view from parts the owner publishes,
//! and continues from the top-left corner rather than from wherever the locked
//! half got to; its line may interleave with another processor's mid-line,
//! which is the same trade the ports make: mangled output beats none.
//!
//! # Scrolling without reading the screen
//!
//! The aperture is uncached device memory, where a store is cheap but a load
//! waits on the bus: moving the picture up a row by copying its pixels reads
//! every byte of the screen back, which on a large panel takes seconds per
//! line. So the characters on screen are kept in [`Text`], in ordinary memory,
//! and a scroll moves them there and redraws only the cells whose character
//! changed — stores, never loads, and few of them on a log whose consecutive
//! lines share most of their prefix. The screen is cleared when the log
//! attaches, so what is on it and what the grid says is on it agree from the
//! start. A writer that cannot take the grid — an emergency line racing the
//! locked writer — draws without recording, and at the bottom row wraps to the
//! top rather than scrolling, since there is nothing to scroll from.

use core::fmt;

use font8x8::legacy::BASIC_LEGACY;
use handoff::{Channels, Framebuffer};
use spin::Mutex;

/// Pixels each glyph cell is wide and tall on the display.
///
/// One source pixel of the 8×8 glyph becomes a block of this many by this
/// many, which is what keeps an ordinary terminal readable on a panel a
/// modern firmware picks by default.
const SCALE: usize = 2;

/// Glyph width in source pixels, which is the font's own.
const GLYPH_WIDTH: usize = 8;
/// Glyph height in source pixels, which is also the font's own.
const GLYPH_HEIGHT: usize = 8;

/// Cell size on the display, in pixels.
const CELL_WIDTH: usize = GLYPH_WIDTH * SCALE;
/// Cell size on the display, in pixels.
const CELL_HEIGHT: usize = GLYPH_HEIGHT * SCALE;

/// Foreground colour of a drawn glyph: white.
const FOREGROUND: [u8; 3] = [0xFF; 3];
/// Background colour behind and around a drawn glyph: black.
const BACKGROUND: [u8; 3] = [0x00; 3];

/// The lowest character code with a glyph worth drawing.
const FIRST_PRINTABLE: u8 = 0x20;
/// The highest character code in the carried font table.
const LAST_PRINTABLE: u8 = 0x7F;
/// What a character outside the printable range is drawn as.
const UNKNOWN_GLYPH: u8 = b'?';

/// The most cells across the log uses, which covers a 4K panel at this scale.
const MAX_COLUMNS: usize = 256;
/// The most cells down the log uses, which covers a 4K panel at this scale.
const MAX_ROWS: usize = 160;
/// What the grid holds for a cell that shows nothing. The font's glyph for it
/// is empty, so drawing it paints the cell's background, exactly as a space
/// does — which is why a space is recorded as this too.
const BLANK: u8 = 0;

/// The characters on screen, shared by every writer of this image's log.
static TEXT: Mutex<Text> = Mutex::new(Text {
    cells: [[BLANK; MAX_COLUMNS]; MAX_ROWS],
});

/// The characters on screen, one per cell, so a scroll redraws from them
/// rather than reading the frame buffer back.
pub(crate) struct Text {
    /// Each cell's character, by row and then column.
    cells: [[u8; MAX_COLUMNS]; MAX_ROWS],
}

impl Text {
    /// Blanks every cell, in place rather than through a fresh grid, which
    /// would be built on the stack first.
    fn clear(&mut self) {
        self.cells.as_flattened_mut().fill(BLANK);
    }

    /// Records that the cell `cursor` names shows `glyph`.
    fn record(&mut self, cursor: &Cursor, glyph: u8) {
        self.cells[cursor.row][cursor.column] = if glyph == b' ' { BLANK } else { glyph };
    }
}

/// Where the next glyph goes, in cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Cursor {
    /// Cells from the left edge of the display.
    pub(crate) column: usize,
    /// Cells from the top edge of the display.
    pub(crate) row: usize,
}

/// The geometry of a screen, and everything drawing needs besides the bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Canvas {
    /// Visible pixels per scan line.
    pub(crate) width: u32,
    /// Visible scan lines.
    pub(crate) height: u32,
    /// Bytes from one scan line's first pixel to the next.
    pub(crate) pitch: u32,
    /// How a pixel's channels are laid out in those bytes.
    channels: Channels,
}

impl Canvas {
    /// Reads a canvas out of a handoff description, refusing unusable ones.
    pub(crate) fn of(screen: &Framebuffer) -> Option<Self> {
        if !screen.usable() {
            return None;
        }
        let channels = Channels::of(screen.format)?;
        Some(Self {
            width: screen.width,
            height: screen.height,
            pitch: screen.pitch,
            channels,
        })
    }

    /// Bytes the drawing surface spans, which is what a mapping must cover.
    pub(crate) fn span(&self) -> u64 {
        u64::from(self.pitch) * u64::from(self.height)
    }

    /// Bytes from one scan line's first pixel to the next.
    fn pitch_bytes(&self) -> usize {
        usize::try_from(self.pitch).expect("a pitch fits a usize")
    }

    /// Cells across, which is the width less any trailing partial cell, up
    /// to the most the grid holds.
    fn columns(&self) -> usize {
        let cell = u32::try_from(CELL_WIDTH).expect("a cell width fits u32");
        usize::try_from(self.width / cell)
            .expect("cells across fit a usize")
            .min(MAX_COLUMNS)
    }

    /// Cells down, up to the most the grid holds.
    fn rows(&self) -> usize {
        let cell = u32::try_from(CELL_HEIGHT).expect("a cell height fits u32");
        usize::try_from(self.height / cell)
            .expect("cells down fit a usize")
            .min(MAX_ROWS)
    }

    /// Writes one string, starting wherever the cursor says and scrolling
    /// when the bottom row fills.
    ///
    /// Characters the carried font has no answer for draw as
    /// [`UNKNOWN_GLYPH`]; a newline moves to the first cell of the next row,
    /// filling a row that was already the last one by scrolling everything up.
    /// Each glyph is recorded in `grid` when there is one; without it nothing
    /// can be scrolled, and a full display wraps to the top row instead.
    pub(crate) fn write_str(
        &self,
        pixels: &mut [u8],
        cursor: &mut Cursor,
        mut grid: Option<&mut Text>,
        text: &str,
    ) {
        for character in text.chars() {
            let glyph = match u8::try_from(character) {
                Ok(byte @ FIRST_PRINTABLE..=LAST_PRINTABLE) => byte,
                Ok(b'\n') => {
                    self.newline(pixels, cursor, grid.as_deref_mut());
                    continue;
                }
                _ => UNKNOWN_GLYPH,
            };
            self.draw(pixels, cursor, glyph);
            if let Some(grid) = grid.as_deref_mut() {
                grid.record(cursor, glyph);
            }
            cursor.column += 1;
            if cursor.column == self.columns() {
                self.newline(pixels, cursor, grid.as_deref_mut());
            }
        }
    }

    /// Moves to the first cell of the next row, scrolling a full display, or
    /// wrapping it to the top when there is no grid to scroll from.
    fn newline(&self, pixels: &mut [u8], cursor: &mut Cursor, grid: Option<&mut Text>) {
        cursor.column = 0;
        cursor.row += 1;
        if cursor.row == self.rows() {
            match grid {
                Some(grid) => {
                    cursor.row -= 1;
                    self.scroll(pixels, grid);
                }
                None => cursor.row = 0,
            }
        }
    }

    /// Moves every row of the grid up one, blanks the freed bottom row, and
    /// repaints the screen from the grid.
    ///
    /// Scrolling shifts the whole picture up a cell, so nearly every pixel
    /// changes; the repaint writes them in strictly ascending framebuffer
    /// address order (see [`repaint`](Self::repaint)), which the one before it
    /// did not. On the write-combining memory a frame buffer is, ascending
    /// writes coalesce into bursts, where the scattered sixteen-pixel runs a
    /// per-cell redraw emitted did not and cost an order of magnitude more.
    fn scroll(&self, pixels: &mut [u8], grid: &mut Text) {
        let rows = self.rows();
        for row in 0..rows {
            for column in 0..self.columns() {
                grid.cells[row][column] = if row + 1 < rows {
                    grid.cells[row + 1][column]
                } else {
                    BLANK
                };
            }
        }
        self.repaint(pixels, grid);
    }

    /// Repaints the whole screen from the grid, one scan line at a time from
    /// top to bottom.
    ///
    /// Each scan line's pixels are composed once into a buffer in ordinary
    /// cached memory and then copied to the frame buffer in a single
    /// [`copy_from_slice`](slice::copy_from_slice) — a `memcpy`, which the
    /// frame buffer's write-combining memory turns into burst writes. The
    /// scattered sixteen-pixel runs a per-cell redraw emitted left that
    /// memory's few fill buffers thrashing, and cost an order of magnitude
    /// more. Because the `SCALE` scan lines a glyph row covers are
    /// identical, the buffer is built once per glyph row and copied to
    /// each.
    fn repaint(&self, pixels: &mut [u8], grid: &Text) {
        let ink = u32::from_ne_bytes(self.channels.encode(FOREGROUND));
        let blank = u32::from_ne_bytes(self.channels.encode(BACKGROUND));
        let stride = self.pitch_bytes() / Framebuffer::BYTES_PER_PIXEL as usize;
        let columns = self.columns();
        let width = columns * CELL_WIDTH;
        // SAFETY: the frame buffer begins on a page boundary and its pitch is a
        // whole number of pixels, so the bytes are aligned for `u32` and the
        // prefix is empty; every bit pattern is a valid `u32`, and these bytes
        // are initialized device memory this writer owns for the call.
        let (_, words, _) = unsafe { pixels.align_to_mut::<u32>() };
        let mut line = [blank; MAX_COLUMNS * CELL_WIDTH];
        for cell_row in 0..self.rows() {
            for font_row in 0..GLYPH_HEIGHT {
                for cell_column in 0..columns {
                    let glyph = BASIC_LEGACY[usize::from(grid.cells[cell_row][cell_column])];
                    let bits = glyph[font_row];
                    let origin = cell_column * CELL_WIDTH;
                    for bit in 0..GLYPH_WIDTH {
                        let colour = if bits & (1 << bit) != 0 { ink } else { blank };
                        let pixel = origin + bit * SCALE;
                        line[pixel..pixel + SCALE].fill(colour);
                    }
                }
                for dy in 0..SCALE {
                    let y = cell_row * CELL_HEIGHT + font_row * SCALE + dy;
                    let start = y * stride;
                    words[start..start + width].copy_from_slice(&line[..width]);
                }
            }
        }
    }

    /// Draws one glyph into the cell the cursor names, background included,
    /// so a cell never shows a blend of this line and the last one.
    fn draw(&self, pixels: &mut [u8], cursor: &Cursor, byte: u8) {
        let ink = self.channels.encode(FOREGROUND);
        let blank = self.channels.encode(BACKGROUND);
        let origin_x = cursor.column * CELL_WIDTH;
        let origin_y = cursor.row * CELL_HEIGHT;
        for (row_index, row) in BASIC_LEGACY[usize::from(byte)].into_iter().enumerate() {
            for bit in 0..GLYPH_WIDTH {
                // The font spells rows top-down with bit zero leftmost.
                let colour = if row & (1 << bit) != 0 { ink } else { blank };
                for dy in 0..SCALE {
                    for dx in 0..SCALE {
                        self.write_pixel(
                            pixels,
                            origin_x + bit * SCALE + dx,
                            origin_y + row_index * SCALE + dy,
                            colour,
                        );
                    }
                }
            }
        }
    }

    /// Stores one pixel, ignoring coordinates off the visible area rather
    /// than wrapping them into a neighbouring scan line.
    fn write_pixel(&self, pixels: &mut [u8], x: usize, y: usize, colour: [u8; 4]) {
        let width = usize::try_from(self.width).expect("a width fits a usize");
        let height = usize::try_from(self.height).expect("a height fits a usize");
        if x >= width || y >= height {
            return;
        }
        // Widening: the constant is `u32`, and `usize` carries no lossless
        // conversion from it on this target.
        let offset = y * self.pitch_bytes() + x * Framebuffer::BYTES_PER_PIXEL as usize;
        pixels[offset..offset + colour.len()].copy_from_slice(&colour);
    }
}

/// The frame buffer as a writer: the geometry, the address its bytes live at,
/// and the cursor between lines.
pub(crate) struct Efifb {
    /// Address bytes are written through, which the mapper chose.
    pub(crate) address: u64,
    /// The screen's geometry.
    pub(crate) canvas: Canvas,
    /// Where the next glyph goes.
    pub(crate) cursor: Cursor,
}

impl Efifb {
    /// Builds the backend over a handoff description, writing through
    /// `address`.
    ///
    /// The address must be the first writable byte of exactly
    /// [`Canvas::span`] bytes — the caller mapped the described framebuffer
    /// there, and nothing unmaps or retargets that translation for as long as
    /// the machine runs, which is what makes every later slice sound.
    pub(crate) fn describe(screen: &Framebuffer, address: u64) -> Option<Self> {
        Some(Self {
            address,
            canvas: Canvas::of(screen)?,
            cursor: Cursor::default(),
        })
    }

    /// Blanks the whole screen and the grid, so the two agree before the
    /// first line is drawn, and moves the cursor to the top-left cell.
    ///
    /// Whatever firmware or the loader left on the screen is not in the grid,
    /// and a scroll redraws only the cells the grid says changed — so anything
    /// not cleared here would stay on screen behind the log.
    pub(crate) fn clear(&mut self) {
        TEXT.lock().clear();
        // SAFETY: as in `write_str`; the bytes are filled and the slice
        // dropped before this returns.
        unsafe { pixels(self.address, &self.canvas) }.fill(0);
        self.cursor = Cursor::default();
    }
}

impl fmt::Write for Efifb {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let Efifb {
            address,
            canvas,
            cursor,
        } = self;
        // SAFETY: `address` was vouched for by whoever built this writer
        // through `describe`: it names the first byte of a live mapping
        // covering `canvas.span()` bytes, established once and never moved,
        // so the slice cannot leave it. Writers are serialized by the crate's
        // output lock except the emergency path, whose interleaving is
        // accepted where it is used; and no reference into these bytes
        // outlives this call.
        let pixels = unsafe { pixels(*address, canvas) };
        // Taken without waiting: the locked writer is the only one that takes
        // it in ordinary running, and an emergency line that finds it held
        // draws unrecorded rather than wait on a lock its own processor may
        // hold.
        let mut grid = TEXT.try_lock();
        canvas.write_str(pixels, cursor, grid.as_deref_mut(), s);
        Ok(())
    }
}

/// The frame buffer's bytes, as a slice over the mapping at `address`.
///
/// # Safety
///
/// `address` must name the first byte of a live, writable mapping covering
/// `canvas.span()` bytes that nothing unmaps while the slice is in use, and the
/// slice must not outlive the write it is taken for.
unsafe fn pixels<'a>(address: u64, canvas: &Canvas) -> &'a mut [u8] {
    let span = usize::try_from(canvas.span()).expect("the span fits a usize");
    // SAFETY: the caller guarantees the mapping and how long the slice lives.
    unsafe { core::slice::from_raw_parts_mut(address as *mut u8, span) }
}

#[cfg(test)]
mod tests {
    //! Drawing decisions, over buffers that stand in for the mapping.

    use handoff::Channels;

    use super::{
        BACKGROUND, CELL_HEIGHT, CELL_WIDTH, Canvas, Cursor, FOREGROUND, GLYPH_HEIGHT, SCALE, TEXT,
        Text,
    };
    extern crate std;
    use std::{vec, vec::Vec};

    use spin::MutexGuard;

    /// The image's grid, blanked, held for the rest of the test so tests that
    /// draw through it take turns.
    fn grid() -> MutexGuard<'static, Text> {
        let mut grid = TEXT.lock();
        grid.clear();
        grid
    }

    /// A screen of two cells across and two down, with no padding.
    fn small_canvas() -> Canvas {
        let cell_width = u32::try_from(CELL_WIDTH).expect("cell width fits");
        let cell_height = u32::try_from(CELL_HEIGHT).expect("cell height fits");
        Canvas {
            width: 2 * cell_width,
            height: 2 * cell_height,
            pitch: 2 * cell_width * handoff::Framebuffer::BYTES_PER_PIXEL,
            channels: Channels::RedGreenBlue,
        }
    }

    /// A buffer standing in for the mapping of [`small_canvas`].
    fn small_pixels() -> Vec<u8> {
        vec![0xEE; small_canvas().span().try_into().expect("span fits")]
    }

    /// Reads the channel words of the pixel at `(x, y)` as red, green, blue.
    fn pixel(pixels: &[u8], x: usize, y: usize) -> [u8; 3] {
        let pitch = usize::try_from(small_canvas().pitch).expect("pitch fits");
        let offset = y * pitch + x * 4;
        [pixels[offset + 2], pixels[offset + 1], pixels[offset]]
    }

    /// Coordinates of every inked pixel, found by scanning for foreground.
    fn inked(pixels: &[u8]) -> Vec<(usize, usize)> {
        let canvas = small_canvas();
        let width = usize::try_from(canvas.width).expect("width fits");
        let height = usize::try_from(canvas.height).expect("height fits");
        let mut found = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if pixel(pixels, x, y) == FOREGROUND {
                    found.push((x, y));
                }
            }
        }
        found
    }

    #[test]
    fn a_description_is_read_only_when_it_is_usable() {
        let usable = handoff::Framebuffer {
            base: 0xF000_0000,
            pitch: 7680,
            width: 1920,
            height: 1080,
            format: handoff::Framebuffer::BGRX,
        };
        assert!(Canvas::of(&usable).is_some());

        let blt_only = handoff::Framebuffer {
            format: 3,
            ..usable
        };
        assert!(Canvas::of(&blt_only).is_none());

        let short_pitch = handoff::Framebuffer {
            pitch: 1919 * 4,
            ..usable
        };
        assert!(Canvas::of(&short_pitch).is_none());
    }

    #[test]
    fn the_span_covers_every_scan_line_including_the_padding() {
        let padded = Canvas {
            pitch: 40 * 4,
            ..small_canvas()
        };
        assert_eq!(padded.span(), 40 * 4 * 32);
    }

    #[test]
    fn a_glyph_draws_its_pixels_and_the_background_fills_the_rest_of_the_cell() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        canvas.draw(&mut pixels, &Cursor::default(), b'|');
        // Inside the cell every pixel is one of the two colours — never the
        // stand-in pattern, which is what a background-less draw would leave
        // behind around the glyph.
        for y in 0..CELL_HEIGHT {
            for x in 0..CELL_WIDTH {
                let seen = pixel(&pixels, x, y);
                assert!(
                    seen == FOREGROUND || seen == BACKGROUND,
                    "pixel ({x}, {y}) is neither ink nor background"
                );
            }
        }
        // And outside it nothing was touched.
        assert_eq!(pixel(&pixels, CELL_WIDTH + 1, CELL_HEIGHT + 1), [0xEE; 3]);
    }

    #[test]
    fn one_source_pixel_becomes_a_block_of_the_scale() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        canvas.draw(&mut pixels, &Cursor::default(), b'|');
        let lit = inked(&pixels);
        assert!(!lit.is_empty(), "the glyph drew nothing");
        // Every lit source pixel appears as SCALE×SCALE screen pixels: each
        // lit position has its whole block beside it.
        for (x, y) in &lit {
            let source_x = x / SCALE;
            let source_y = y / SCALE;
            for dy in 0..SCALE {
                for dx in 0..SCALE {
                    assert_eq!(
                        pixel(&pixels, source_x * SCALE + dx, source_y * SCALE + dy),
                        FOREGROUND,
                        "block at ({source_x}, {source_y}) incomplete"
                    );
                }
            }
        }
    }

    #[test]
    fn the_channel_order_decides_which_byte_leads_the_pixel() {
        let rgbx = small_canvas();
        let bgrx = Canvas {
            channels: Channels::BlueGreenRed,
            ..rgbx
        };
        let mut left = small_pixels();
        let mut right = small_pixels();
        rgbx.draw(&mut left, &Cursor::default(), b'|');
        bgrx.draw(&mut right, &Cursor::default(), b'|');
        // White reads identically in both orders, so the first byte of an inked
        // pixel names the layout: red under one spelling, blue under the other.
        let first_red = inked(&left)[0];
        assert_eq!(pixel(&left, first_red.0, first_red.1)[0], 0xFF);
        let first_blue = inked(&right)[0];
        assert_eq!(pixel(&right, first_blue.0, first_blue.1)[2], 0xFF);
        assert_eq!((first_red.0, first_red.1), (first_blue.0, first_blue.1));
    }

    #[test]
    fn a_newline_moves_to_the_first_cell_of_the_next_row() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        let mut cursor = Cursor::default();
        canvas.write_str(&mut pixels, &mut cursor, Some(&mut grid()), "\n");
        assert_eq!(cursor, Cursor { column: 0, row: 1 });
    }

    #[test]
    fn filling_the_last_cell_of_a_row_starts_the_next_one() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        let mut cursor = Cursor::default();
        canvas.write_str(&mut pixels, &mut cursor, Some(&mut grid()), "..");
        assert_eq!(cursor.column, 0);
        assert_eq!(cursor.row, 1);
    }

    #[test]
    fn a_newline_on_the_last_row_scrolls_one_cell_height_and_clears_behind_it() {
        let canvas = small_canvas();
        let mut pixels = vec![0; small_pixels().len()];
        let mut grid = grid();
        let last_row = canvas.rows() - 1;
        let mut cursor = Cursor {
            column: 0,
            row: last_row,
        };
        canvas.write_str(&mut pixels, &mut cursor, Some(&mut grid), "#");
        let drawn = inked(&pixels);
        assert!(!drawn.is_empty());

        canvas.newline(&mut pixels, &mut cursor, Some(&mut grid));

        assert_eq!(cursor.row, last_row);
        // Every inked pixel moved up exactly one cell height and no ink is
        // left where it was.
        for (x, y) in drawn {
            assert_eq!(pixel(&pixels, x, y - CELL_HEIGHT), FOREGROUND);
            assert_eq!(pixel(&pixels, x, y), BACKGROUND);
        }
    }

    #[test]
    fn a_scroll_shifts_the_grid_up_and_repaints_from_it() {
        // A glyph in the grid's bottom row, present nowhere in the pixels: a
        // scroll moves it up a row in the grid, blanks the freed bottom row,
        // and repaints the whole screen from the grid — so the glyph that was
        // only recorded appears one cell-row up and the old bottom row clears.
        let canvas = small_canvas();
        let mut pixels = vec![0; small_pixels().len()];
        let mut grid = grid();
        let last_row = canvas.rows() - 1;
        grid.cells[last_row][0] = b'#';
        // The cursor on the last row, so the newline below scrolls.
        let mut cursor = Cursor {
            column: 0,
            row: last_row,
        };

        canvas.newline(&mut pixels, &mut cursor, Some(&mut grid));

        assert_eq!(cursor.row, last_row);
        assert_eq!(
            grid.cells[last_row - 1][0],
            b'#',
            "the grid did not shift up"
        );
        assert_eq!(grid.cells[last_row][0], 0, "the freed row is not blank");
        let lit = inked(&pixels);
        assert!(!lit.is_empty(), "the repaint drew nothing from the grid");
        assert!(
            lit.iter().all(|(_, y)| *y < CELL_HEIGHT),
            "the glyph was not repainted one row up"
        );
    }

    #[test]
    fn without_a_grid_a_full_display_wraps_to_the_top_row() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        let mut cursor = Cursor {
            column: 0,
            row: canvas.rows() - 1,
        };
        canvas.write_str(&mut pixels, &mut cursor, None, "\n");
        assert_eq!(cursor, Cursor::default());
    }

    #[test]
    fn characters_without_glyphs_draw_as_the_unknown_mark() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        let mut cursor = Cursor::default();
        canvas.write_str(&mut pixels, &mut cursor, Some(&mut grid()), "\u{2603}");
        // The unknown mark drew into the first cell, which is what disturbing
        // its stand-in pattern proves; the point is that nothing panicked and
        // nothing was skipped.
        assert!(pixels[..CELL_WIDTH * 4].iter().any(|byte| *byte != 0xEE));
        assert_eq!(cursor.column, 1);
    }

    #[test]
    fn glyphs_are_eight_rows_tall_so_a_cell_is_scale_times_that() {
        // Pinned because the scroll distance and the cell grid both derive
        // from it; the constant lives in the font's shape, not ours.
        assert_eq!(GLYPH_HEIGHT, 8);
        assert_eq!(CELL_HEIGHT, SCALE * GLYPH_HEIGHT);
    }
}
