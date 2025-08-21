# pymovements Toy Dataset

This repository serves as an example toy dataset resource for the
[pymovements](https://github.com/aeye-lab/pymovements) package.

It features gaze data from a single subject reading 4 texts with 5 screens each.

Filenames have the format `trial_{text_id}_{page_id}.csv`.

The experiment configuration is specified as:
```
Experiment(
	screen_width_px=1280,
	screen_height_px=1024,
	screen_width_cm=38,
	screen_height_cm=30.2,
	distance_cm=68,
	origin='lower left',
	sampling_rate=1000,
)
```