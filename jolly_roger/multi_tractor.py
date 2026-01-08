from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
from tqdm.auto import tqdm

from jolly_roger.logging import logger
from jolly_roger.tractor import (
    TukeyTractorOptions,
    TukeyTractorResults,
    _tukey_tractor,
    add_output_column,
    get_data_chunks,
    get_open_ms_tables,
    make_plot_results,
)
from jolly_roger.utils import log_dataclass_attributes, log_jolly_roger_version
from jolly_roger.uvws import WDelays, get_object_delay_for_ms


@dataclass
class MultiTukeyTractorOptions:
    """Options to describe the tukey taper to apply"""

    ms_path: Path
    """Measurement set to be modified"""
    target_objects: list[str]
    """The target objects to apply the delay towards."""
    outer_width_ns: float = 10
    """The start of the tapering in nanoseconds"""
    tukey_width_ns: float = 10
    """The width of the tapered region in nanoseconds"""
    data_column: str = "DATA"
    """The visibility column to modify"""
    output_column: str = "CORRECTED_DATA"
    """The output column to be created with the modified data"""
    copy_column_data: bool = False
    """Copy the data from the data column to the output column before applying the taper"""
    dry_run: bool = False
    """Indicates whether the data will be written back to the measurement set"""
    make_plots: bool = False
    """Create a small set of diagnostic plots. This can be slow."""
    overwrite: bool = False
    """If the output column exists it will be overwritten"""
    chunk_size: int = 1000
    """Size of the row-wise chunking iterator"""
    elevation_cut: u.Quantity = -1 * u.deg
    """The elevation cut-off for the target object. Defaults to 0 degrees."""
    ignore_nyquist_zone: int = 2
    """Do not apply the tukey taper if object is beyond this Nyquist zone"""
    reverse_baselines: bool = False
    """Reverse baseline ordering"""


def tukey_multi_tractor(
    multi_tukey_tractor_options: MultiTukeyTractorOptions,
) -> TukeyTractorResults:
    """Iterate row-wise over a specified measurement set and
    apply a tukey taper operation to the delay data. Iteration
    is performed based on a chunk soize, indicating the number
    of rows to read in at a time.

    Full description of options are outlined in `TukeyTaperOptions`.

    Args:
        tukey_tractor_options (TukeyTractorOptions): The settings to use during the taper, and measurement set to apply them to.

    Returns:
        TukeyTractorResults: Representative information of the tapering process
    """
    log_jolly_roger_version()
    log_dataclass_attributes(
        to_log=multi_tukey_tractor_options, class_name="TukeyTaperOptions"
    )

    # acquire all the tables necessary to get unit information and data from
    open_ms_tables = get_open_ms_tables(
        ms_path=multi_tukey_tractor_options.ms_path, read_only=False
    )

    if not multi_tukey_tractor_options.dry_run:
        add_output_column(
            tab=open_ms_tables.main_table,
            output_column=multi_tukey_tractor_options.output_column,
            data_column=multi_tukey_tractor_options.data_column,
            overwrite=multi_tukey_tractor_options.overwrite,
            copy_column_data=multi_tukey_tractor_options.copy_column_data,
        )

    w_delays_dict = dict[str, WDelays]()
    for target_object in multi_tukey_tractor_options.target_objects:
        # Generate the delay for all baselines and time steps
        logger.info(f"Pre-calculating delays towards the target: {target_object}")
        w_delays_dict[target_object] = get_object_delay_for_ms(
            ms_path=multi_tukey_tractor_options.ms_path,
            object_name=target_object,
            reverse_baselines=multi_tukey_tractor_options.reverse_baselines,
        )

    if not multi_tukey_tractor_options.dry_run:
        with tqdm(total=len(open_ms_tables.main_table)) as pbar:
            for data_chunk in get_data_chunks(
                open_ms_tables=open_ms_tables,
                chunk_size=multi_tukey_tractor_options.chunk_size,
                data_column=multi_tukey_tractor_options.data_column,
            ):
                for target_object in multi_tukey_tractor_options.target_objects:
                    # TODO: Clean up this doubling of options - this is prone to sinking a ship
                    _tukey_tractor_options = TukeyTractorOptions(
                        ms_path=multi_tukey_tractor_options.ms_path,
                        outer_width_ns=multi_tukey_tractor_options.outer_width_ns,
                        tukey_width_ns=multi_tukey_tractor_options.tukey_width_ns,
                        data_column=multi_tukey_tractor_options.data_column,
                        output_column=multi_tukey_tractor_options.output_column,
                        copy_column_data=multi_tukey_tractor_options.copy_column_data,
                        dry_run=multi_tukey_tractor_options.dry_run,
                        make_plots=multi_tukey_tractor_options.make_plots,
                        overwrite=multi_tukey_tractor_options.overwrite,
                        chunk_size=multi_tukey_tractor_options.chunk_size,
                        target_object=target_object,
                        apply_towards_object=True,
                        ignore_nyquist_zone=multi_tukey_tractor_options.ignore_nyquist_zone,
                        reverse_baselines=multi_tukey_tractor_options.reverse_baselines,
                    )
                    taper_data_chunk, flags_to_apply = _tukey_tractor(
                        data_chunk=data_chunk,
                        tukey_tractor_options=_tukey_tractor_options,
                        w_delays=w_delays_dict[target_object],
                    )

                    pbar.update(len(taper_data_chunk.masked_data))

                    # only put if not a dry run
                    open_ms_tables.main_table.putcol(
                        columnname=multi_tukey_tractor_options.output_column,
                        value=taper_data_chunk.masked_data,
                        startrow=taper_data_chunk.row_start,
                        nrow=taper_data_chunk.chunk_size,
                    )
                    if flags_to_apply is not None:
                        open_ms_tables.main_table.putcol(
                            columnname="FLAG",
                            value=flags_to_apply,
                            startrow=taper_data_chunk.row_start,
                            nrow=taper_data_chunk.chunk_size,
                        )

    if multi_tukey_tractor_options.make_plots:
        plot_paths: list[Path] | None = []
        for target_object in multi_tukey_tractor_options.target_objects:
            plot_paths.extend(  # type: ignore[union-attr]
                make_plot_results(
                    open_ms_tables=open_ms_tables,
                    data_column=multi_tukey_tractor_options.data_column,
                    output_column=multi_tukey_tractor_options.output_column,
                    target=target_object,
                    w_delays=w_delays_dict[target_object],
                    reverse_baselines=multi_tukey_tractor_options.reverse_baselines,
                    outer_width_ns=multi_tukey_tractor_options.outer_width_ns,
                )
            )

        logger.info(f"Made {len(plot_paths)} output plots")  # type: ignore[arg-type]
    else:
        plot_paths = None

    return TukeyTractorResults(
        ms_path=open_ms_tables.ms_path,
        output_column=multi_tukey_tractor_options.output_column,
        output_plots=plot_paths,
    )


def get_parser() -> ArgumentParser:
    """Create the CLI argument parser

    Returns:
        ArgumentParser: Constructed argument parser
    """
    parser = ArgumentParser(
        description="Run the Jolly Roger Tractor",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode")

    tukey_parser = subparsers.add_parser(
        name="tukey", help="Perform a simple Tukey taper across delay-time data"
    )
    tukey_parser.add_argument(
        "ms_path",
        type=Path,
        help="The measurement set to process with the Tukey tractor",
    )
    tukey_parser.add_argument(
        "--outer-width",
        type=float,
        default=10,
        help="The outer width of the Tukey taper in nanoseconds",
    )
    tukey_parser.add_argument(
        "--tukey-width",
        type=float,
        default=5,
        help="The Tukey width of the Tukey taper in nanoseconds",
    )
    tukey_parser.add_argument(
        "--data-column",
        type=str,
        default="DATA",
        help="The data column to use for the Tukey tractor",
    )
    tukey_parser.add_argument(
        "--output-column",
        type=str,
        default="CORRECTED_DATA",
        help="The output column to write the Tukey tractor results to",
    )
    tukey_parser.add_argument(
        "--copy-column-data",
        action="store_true",
        help="If set, the Tukey tractor will copy the data from the data column to the output column before applying the taper",
    )
    tukey_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, the Tukey tractor will not write any output, but will log what it would do",
    )
    tukey_parser.add_argument(
        "--make-plots",
        action="store_true",
        help="If set, the Tukey tractor will make plots of the results. This can be slow.",
    )
    tukey_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, the Tukey tractor will overwrite the output column if it already exists",
    )
    tukey_parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="The number of rows to process in one chunk. Larger numbers require more memory but fewer interactions with I/O.",
    )
    tukey_parser.add_argument(
        "--ignore-nyquist-zone",
        type=int,
        default=2,
        help="Do not apply the taper if the objects delays beyond this Nyquist zone",
    )
    tukey_parser.add_argument(
        "--reverse-baselines",
        action="store_true",
        help="Reverse baseline ordering",
    )
    tukey_parser.add_argument(
        "--target-objects",
        type=str,
        default="Sun",
        nargs="+",
        help="The target objects to apply the delay towards. e.g. --target-objects Sun CenA CygA",
    )

    return parser


def cli() -> None:
    """Command line interface for the Jolly Roger Tractor."""
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == "tukey":
        multi_tukey_tractor_options = MultiTukeyTractorOptions(
            ms_path=args.ms_path,
            outer_width_ns=args.outer_width,
            tukey_width_ns=args.tukey_width,
            data_column=args.data_column,
            output_column=args.output_column,
            copy_column_data=args.copy_column_data,
            dry_run=args.dry_run,
            make_plots=args.make_plots,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
            target_objects=args.target_objects,
            ignore_nyquist_zone=args.ignore_nyquist_zone,
            reverse_baselines=args.reverse_baselines,
        )

        tukey_multi_tractor(multi_tukey_tractor_options=multi_tukey_tractor_options)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
