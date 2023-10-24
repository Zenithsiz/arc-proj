# Arc Proj

> This repository is part of a project submitted for a university class ARC (Network science - 2023/2024 1st semester) by group 7.
>
> Authors:
>
> - 96735: Filipe Rodrigues
> - 96766: Sara Sant’Ana
> - 97366: Francisco Silva
>
> See the `proj` branch for the code submitted for the project.

# Configuration

Before running, you must create a python venv and install the requirements in `requirements.txt`.

For example (if using bash):

```bash
# Create the virtual environment
python3 -m venv .venv

# Update the current shell with the environment binaries
source .venv/bin/activate

# Install the requirements on the environment.
pip3 install -r requirements.txt
```

Note that if you're not using bash, there are other `activate` scripts to use. See the directory `.venv/bin/` for other activation scripts.

# Usage

You may run the simulation using

```bash
python src/main.py
```

To configure it, there are several variables defined in the script you may modify:

## Execution method

Within `main`, you have `exec_method`, which dictates how the simulation is going to be run.

### `ExecMethod.NORMAL`

This mode simply runs the simulation once and quits.

It is intended to be used while debugging or showcasing a single execution.

The `params` variable dictates the parameters for the run (See [Run parameters](<#Run\ parameters>))

### `ExecMethod.BENCHMARK`

This mode runs the simulation several times, recording samples of how long each run took.

It can be further configured via the following variables:

- `max_samples`: Max number of samples to take
- `max_time_s`: Max time to run the benchmark (across all runs) in seconds.

The `params` variable dictates the parameters for the run (See [Run parameters](<#Run\ parameters>) for more details)

Note that for this method, the display method should be `DisplayMethod.NONE` to avoid benchmarking the slow display code (See [Display method](<#Display\ method>) for more details).

### `ExecMethod.JSON_OUTPUT`

This mode runs the simulation multiple times with a different seed, outputting the run results into a json file named after that seed.

It can be further configured via the following variables:

- `seed_start`: The first seed to test
- `seed_end`: One past the last seed to test

All in all, all seeds in `range(seed_start, seed_end)` are run.

Note that for this method, the display method should be `DisplayMethod.NONE` to avoid having to close the window after each run (See [Display method](<#Display\ method>) for more details).

## Run parameters

The simulation takes a set of parameters to determine how it functions:

### `graph_size`

This tuple determines the graph size. A typical value would be 80x80, which is large enough to notice the complexity of the simulation, but small enough that it runs very quickly.

A recommended limit would be 1000x1000 or 2000x2000, which consume, respectively, ~1.5 GiB and ~6.0 GiB of memory.

### `seed`

This is the seed that governs all random decisions taken by the simulation.

You may pass `None` to generate a random seed.

### `empty_chance`

This is a value that determines how likely it is for any node to be empty.

A normal value used would be `0.1` (10%) or `0.2` (20%).

### `agent_weights`

This is a dictionary of weights that affects how likely it is for any single agent to be chosen.

The weights will be normalized, so you may use any scale you see fit for defining them.

See [Agents](#Agents) for more details on agents.

### `output_json_path`

Each run may output statistics to a json file. If this value isn't `None`, the run will write these statistics to it.

Currently the statistics contain the following:

- `average_satisfactions: list[float]`: A list of all the average satisfactions for each round of the simulation.

### `output_img_agent_path` / `output_img_satisfaction_path`

The images that are output when using `DisplayMethod.GRID` may be instead redirected to file using these parameters if using `DisplayMethod.GRID_FILE` (See [Display method](<#Display\ method>) for more details).

The first one is the output of the agent view (left) and the second is the output of the satisfaction view (right).

### `display_method`

See [Display method](<#Display\ method>) for more details.

### `rounds_per_display`

Since displaying the graph to the screen is a slow operation, you may choose to perform multiple rounds per each display.

Once equilibrium is reached, the simulation will be drawn regardless of this parameter.

## Display method

There are several ways to visualize the output of the simulation, governed by the `display_method` field of `RunParams` (See [Run parameters](<#Run\ Parameters>) for more details).

### `DisplayMethod.NONE`

This mode doesn't output anything to screen. It is ideal when using `ExecMethod.BENCHMARK` / `ExecMethod.JSON_OUTPUT`, or when one only needs the information printed by the simulation.

### `DisplayMethod.GRAPH`

This mode displays the graph as a ... graph, with the nodes and edges.

It is mostly useful when modifying the edges between each node or just to confirm which edges exist.

### `DisplayMethod.GRID`

This method displays the whole graph as a 2d grid with all agents on it.

It is the easiest way to visualize the graph, especially on higher sizes.

### `DisplayMethod.GRID_FILE`

Similar to `DisplayMethod.GRID`, but outputs the images to a file. (See [Run parameters](<#Run\ Parameters>) for more details, in particular `output_img_agent_path` and `output_img_satisfaction_path`)

## Agents

There are 2 agent kinds provided in this simulation:

### `NAgent`

This is the agent used in the original model.

It has a field, `kind` with two possible values, `NAgentKind.RED` and `NAgentKind.BLUE`.

It calculates satisfaction by checking how many equal neighbors it has, as a percentage of it's total number of neighbors.

If it has no neighbors, it is considered fully satisfied.

### `GAgent`

This is the agent that we've used to expand the original model.

It has a field, `inner`, of type `float`, with values ranging from `0.0` to `1.0` (inclusive).

It calculates satisfaction by performing a weighted sum over all neighbors of their `inner` fields' distance.

If it has no neighbors, it is considered fully satisfied.
