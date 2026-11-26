# Delay CLEAN

`jolly_tractor clean` removes bright objects outside the field by CLEAN in delay, as an alternative to the notch. It needs the optional `rm-lite` dependency: `pip install 'jolly-roger[clean]'`.

```bash
jolly_tractor clean my.ms --target-objects sun --output-column CORRECTED_DATA
```

## How it works

- Each visibility spectrum is transformed to delay with an FFT. The delay spread function (the delay spectrum of a point source) is the FFT of the channel weights, so flagged channels are accounted for exactly.
- A Hogbom CLEAN (`rm_lite.utils.clean.minor_loop_batch`) runs over the whole delay axis. Its mask starts at the brightest emission and grows into connected emission above the noise, so the object's size, structure and spectrum need not be known.
- CLEAN runs in rm-lite's compiled loop over batches of spectra that share a flag pattern, so `--max-workers` threads scale.
- Only components outside the field's delay range are subtracted, so the field is not modified. The subtracted model is evaluated in every channel, including flagged ones.
- Rows where the subtracted emission runs into the field's delay range are flagged, because there the object and field cannot be separated.
- The noise sets the CLEAN mask and threshold. Unless it is given, it is estimated from each spectrum by `estimate_visibility_noise`: each long run of unflagged channels is windowed and transformed to delay, and a low quantile of the delay amplitudes is compared with the Rayleigh distribution of noise. Emission only raises the estimate, so it errs high (by about 20% next to an object 10^4 times brighter than the noise).
- Only rows where an object in `--target-objects` is above the elevation cut are cleaned. The field's delay range on each baseline is `--field-delay-ns` plus the guard band, which is sized from the field of view. The guard band is on by default for both `tukey` and `clean`; `--no-guard-field` turns it off.
- The run is summarised in the log and the measurement set's HISTORY table: how many spectra were cleaned and rows flagged; how bright a spectral line in the field must be before it is partly subtracted; and the residual left at the band edges compared with mid-band.

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
