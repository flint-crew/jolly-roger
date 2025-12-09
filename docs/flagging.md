# Projected baseline flagging
## How does it work?

`jolly_roger` will recompute the (u,v,w)-coordinates of a measurement set as if it were tracking the Sun, from which (u,v)-distances are derieved for each baseline and timestep. An updated `FLAG` column can then be inserted into the measurement set suppressing visibilities that would be sensitive to a nominated range of angular scales. This mode also allows for flagging based on the elevation of the target source.

## Example

`jolly_roger` has a CLI entry point for projected baseline flagging that can be called as:

```
jolly_flagger scienceData.EMU_1141-55.SB47138.EMU_1141-55.beam00_averaged_cal.leakage.ms --min-horizon-limit-deg '-2' --max-horizon-limit-deg 30 --min-scale-deg 0.075
```

Here we are flagging visibilities that correspond to instances where:
- the Sun has an elevation between -2 and 30 degrees, and
- they are sensitive to angular scales larger than 0.075 degrees.

## CLI

```{argparse}
:ref: jolly_roger.flagger.get_parser
:prog: jolly_flagger
```
