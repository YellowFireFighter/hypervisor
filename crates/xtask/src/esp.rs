//! Builds the UEFI crates and stages the EFI system partition.
//!
//! The ESP is a plain directory rather than a filesystem image: QEMU's
//! virtual-FAT driver serves it to the guest directly, so boot media is
//! regenerated from source on every run and no image artifact ever needs to
//! be built, tracked, or cleaned up.

use std::{env, fs, path::PathBuf, process::Command};

use anyhow::{Context, Result};

use crate::{paths, proc};

/// UEFI crates staged into the ESP: package name, image file produced under
/// `target/`, and destination path inside the ESP.
const IMAGES: [(&str, &str, &str); 2] = [
    ("hv-loader", "hv-loader.efi", "EFI/BOOT/BOOTX64.EFI"),
    ("hv-core", "hv-core.efi", "citrine.efi"),
];

/// The boot manager the loader starts as the first guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    /// The Windows boot manager, found on the guest's own system partition.
    Windows,
    /// Limine, which the Linux demonstration disks carry.
    Limine,
}

/// What both images show on the screen while the host comes up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    /// The boot screen: the mark, its shine, the loading bar and the spinner.
    Splash,
    /// The log, drawn line by line as it is written.
    Log,
}

impl Screen {
    /// The feature both images build this with.
    const fn feature(self) -> &'static str {
        match self {
            Self::Splash => "splash",
            Self::Log => "efifb",
        }
    }
}

/// Compiles the UEFI crates and repopulates `dist/esp/` from scratch,
/// returning its path.
///
/// `silent` builds both images with their `quiet` feature, which takes `log`'s
/// static maximum level to `Off` and so compiles every record out of the whole
/// image. It is asked of both packages rather than one, even though `log` is
/// compiled once for the build and either would do it: a flag whose effect
/// depends on feature unification is one that stops working the moment the
/// dependency graph changes.
///
/// `chain` names the boot manager the loader is built to start, which is the
/// loader's own `limine` feature for [`Chain::Limine`], and `screen` what both
/// images draw on the display, asked of both for the same reason `silent` is.
pub fn stage(release: bool, silent: bool, chain: Chain, screen: Screen) -> Result<PathBuf> {
    let root = paths::workspace_root();
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut build = Command::new(cargo);
    build.current_dir(&root).arg("build");
    for (package, _, _) in IMAGES {
        build.args(["--package", package]);
    }
    if release {
        build.arg("--release");
    }
    let mut features: Vec<String> = IMAGES
        .iter()
        .map(|(package, _, _)| format!("{package}/{}", screen.feature()))
        .collect();
    if silent {
        features.extend(
            IMAGES
                .iter()
                .map(|(package, _, _)| format!("{package}/quiet")),
        );
    }
    if chain == Chain::Limine {
        features.push("hv-loader/limine".to_owned());
    }
    build.args(["--features", &features.join(",")]);
    proc::run(&mut build, "it ships with the Rust toolchain")?;

    let profile = if release { "release" } else { "debug" };
    let artifacts = root.join("target/x86_64-unknown-uefi").join(profile);
    let dir = paths::esp_dir();
    if dir.exists() {
        fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to clear stale ESP at {}", dir.display()))?;
    }
    for (_, image, destination) in IMAGES {
        let to = dir.join(destination);
        let parent = to.parent().expect("every ESP destination has a parent");
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        fs::copy(artifacts.join(image), &to)
            .with_context(|| format!("failed to stage {image} as {}", to.display()))?;
    }
    println!("staged ESP at {}", dir.display());
    Ok(dir)
}
