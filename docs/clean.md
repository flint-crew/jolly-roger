# Delay CLEAN

`jolly_tractor clean` removes bright objects outside the field by CLEAN in delay, as an alternative to the notch. It needs the optional `rm-lite` dependency: `pip install 'jolly-roger[clean]'`.

```bash
jolly_tractor clean my.ms --target-objects sun --output-column CORRECTED_DATA
```

## Example

The example below runs `clean` on the same 30 minutes of an ASKAP EMU observation as the [notch example](nulling_example.ipynb). The figures are made from the data each time the documentation is built.

```{toctree}
:maxdepth: 1

clean_example.ipynb
```

## How it works

- Each visibility spectrum is transformed to delay with an FFT. The delay spread function (the delay spectrum of a point source) is the FFT of the channel weights, so flagged channels are accounted for exactly.
- A Hogbom CLEAN (`rm_lite.utils.clean.minor_loop_batch`) runs over the whole delay axis. Its mask starts at the brightest emission and grows into connected emission above the noise, so the object's size, structure and spectrum need not be known.
- CLEAN runs in rm-lite's compiled loop over batches of spectra that share a flag pattern, so `--max-workers` threads scale.
- Only components within each object's delay band, and outside the field's delay range, are subtracted. Other emission, such as other sources or RFI outside the field, is left as it is. The subtracted model is evaluated in every channel, including flagged ones.
- Each object's band is its predicted delay ± (the extent of its disc + D/c + 3/bandwidth):
  - The disc's extent is θ·|(u, v)|/c, with θ the object's angular radius and (u, v) toward the object.
  - D/c covers the spread an object picks up from the frequency-dependent primary beam, seen through its sidelobes. D is the dish diameter: `--dish-diameter-m`, else the largest DISH_DIAMETER. On ASKAP data the Sun's spread is mostly this, ±40 ns, rather than its disc.
  - The last term covers CLEAN's spread of an object between delay samples.
- Each object's radius is set with `--object-radius-deg`, one per `--target-objects` entry, in the same order. By default the Sun is 0.5° (the radio Sun, including the quiet-Sun corona near 1 GHz), the Moon 0.27°, and any other object a point. From Python, set `object_radius_deg` on `CleanTractorOptions`, or pass `object_radii` to `get_object_delay_from_tables`.
- Rows where an object's band meets the field's delay range are flagged, because there the object and field cannot be separated.
- The noise sets the CLEAN mask and threshold. Unless it is given, it is estimated from each spectrum by `estimate_visibility_noise`: each long run of unflagged channels is windowed and transformed to delay, and a low quantile of the delay amplitudes is compared with the Rayleigh distribution of noise. Emission only raises the estimate, so it errs high (by about 20% next to an object 10^4 times brighter than the noise).
- Only rows where an object in `--target-objects` is above the elevation cut are cleaned. The field's delay range on each baseline is `--field-delay-ns` plus the guard band, which is sized from the field of view. The guard band is on by default for both `tukey` and `clean`; `--no-guard-field` turns it off.
- The run is summarised in the log and the measurement set's HISTORY table:
  - each object's radius, and how many spectra were cleaned and rows flagged;
  - the effect on spectral lines (below);
  - the residual left at the band edges compared with mid-band.

## Spectral lines

A narrow spectral line in the field spreads over all delays. It loses the part of that spread that is subtracted. Take a Gaussian line of peak S and FWHM δν, a noise σ per channel, N channels of width Δν, and κ = √(π / 4 ln 2) = 1.06:

- **Below the CLEAN mask the line is untouched.** CLEAN only reaches a line once its delay spectrum, κ S n / N for a line n channels wide, exceeds the mask, `--mask-sigma` times σ/√N. That happens when S/σ > `mask_sigma`·√N/(κ n), which is about 80/n for 288 channels.
- **Brighter lines lose the part of their delay spread inside an object's band** of half-width W at delay τ: a fraction κ δν 2W exp(−(π δν τ)² / 4 ln 2) of the peak, or 2W Δν for a line narrower than a channel. This is the same as for the notch (see [Delay nulling](nulling.md)), and is reported for each object, from the geometry of each baseline.
- **Subtracting everything outside the field, as `clean` did before, removes most of a bright line:** erfc(π δν τ_f / 2√ln 2), or 1 − 2 τ_f Δν for a line narrower than a channel, with τ_f the field's half-width. That is up to 94% for ±30 ns and 1 MHz channels.
- **`--protect-mhz LOW HIGH [LOW HIGH ...]` leaves known lines out of CLEAN altogether.** Those frequency ranges get zero weight, so lines there lose nothing, while the objects are still subtracted from them.
- **The rate step is safe for lines.** A line from the field is steady in time, so it is at the field's fringe rate.

## Recovering flagged rows with fringe-rate CLEAN

`jolly_tractor clean --rate-filter` recovers rows that `clean` flags because the object reaches the field in delay, provided the object is separated from the field in fringe rate. Runs of flagged rows are collected into segments as for the notch's [delay-rate filtering](nulling.md).

- Each channel and polarisation of a segment is CLEANed along time, from the delay-cleaned rows. The response is the FFT of that channel's time weights, so flags are accounted for exactly.
- Only components in each object's predicted fringe-rate band are subtracted, widened by up to 8/T (T is the segment length) but never into the field's. The residual, the field and everything else are left as they are, and weights are not changed.
- A segment is not filtered if an object is within 4/T of the field in fringe rate, and no row is recovered if CLEAN does not converge in any channel.

## Performance

CLEAN costs about 100 times the notch. Throughput in spectra (one row and polarisation) per second, for 288 channels with two bright objects outside the field, on 4 cores:

| | 1 worker | 2 workers | 4 workers |
|---|---|---|---|
| `clean` | 450 | 840 | 1200–1370 |
| `tukey` | 37,000–45,000 | 62,000–70,000 | 94,000–118,000 |

- Use `--max-workers` up to the number of cores. Chunk size does not change throughput, so set `--chunk-size` for memory; there must be at least `--max-workers` chunks for the workers to be used.
- Cost grows in proportion to the number of channels: 148 spectra per second at 2592 channels on 4 workers.
- For a measurement set the size of the bundled test set (2000 rows) reading and writing dominate, and `clean` and `tukey` both take 7–9 s.

Settings that trade quality for speed, with the residual (cleaned minus true, over the noise) on the same data, 4 workers:

| setting | spectra/s | residual |
|---|---|---|
| defaults (`gain=0.3`) | 1360 | 0.37 |
| `gain=0.1` | 1350 | 0.38 |
| `oversample=2` | 1850 | 0.64 |
| `--threshold-sigma 5 --mask-sigma 8` | 1750 | 0.50 |
| `tukey` (for comparison) | 27,000 | 10.2 |

`gain` and `oversample` are set through `DelayCleanOptions` in Python.

## Python interface

`delay_clean_spectrum` works on one spectrum, `delay_clean_spectra` on a stack of spectra that share their channel weights, `compute_delay_clean` on a `DataChunk`, and `compute_clean_tractor` on a `DataChunk` with the objects' predicted delays, as passed to `run_tractor_chunks` by `clean_tractor`.

The options are listed with the other `jolly_tractor` subcommands in the CLI section of [Delay nulling](nulling.md).
