"""
Utilities
"""

import math
from typing import TypeVar, Mapping

import numpy

T = TypeVar('T')
U = TypeVar('U')

def try_index_dict(container: Mapping[U, T], key: U) -> T | None:
	"""
	Gets the value corresponding to the key `key` in `container`, if it exists,
	else returns `None`
	"""

	if key in container:
		return container[key]
	else:
		return None

def fmt_time(s: float) -> str:
	"""
	Formats time in seconds in the closes unit.
	"""
	hours   = s // 3600
	mins    = s // 60   - 60 * hours
	seconds = s // 1    - 60 * mins  - 3600 * hours
	milliseconds = 1e3 * s - 1e0 * int(seconds)
	microseconds = 1e6 * s - 1e3 * int(milliseconds)
	nanoseconds  = 1e9 * s - 1e6 * int(microseconds)

	if hours > 0:
		return f"{hours}h{mins}m{s%60:.2f}s"
	elif mins > 0:
		return f"{mins}m{s%60:.2f}s"
	elif seconds > 0:
		return f"{s:.2f}s"
	elif milliseconds > 0:
		return f"{milliseconds:.2f}ms"
	elif microseconds > 0:
		return f"{microseconds:.2f}μs"
	elif nanoseconds > 0:
		return f"{nanoseconds:.2f}ns"

def reservoir_sample_set(s: set[T], size: int) -> list[T]:
	"""
	Performs a reservoir sample on set `s`, returning a list of length `size`.

	If `s` has less elements than `size`, only that many elements will be returned.

	Uses the optimal algorithm L (from wikipedia).
	It's time complexity if `O(size (1 + log(len(s) / size)))`.
	"""

	# Clamp the size to the length of `s`
	size = min(size, len(s))

	it = s.__iter__()

	# Perform the initial fill
	# Note: `next(it)` will never raise `StopIteration` because
	#        we've clamped the size to the length of `s`.
	output = []
	for _ in range(size):
		output.append(next(it))

	# Then replace the filled elements
	w = math.exp(math.log(numpy.random.random()) / size)
	i = size
	while True:
		i += math.floor( math.log(numpy.random.random()) / math.log(1 - w) ) + 1
		if i >= len(s):
			break

		next_idx = numpy.random.randint(0, size)
		try:
			output[next_idx] = next(it)
		except StopIteration:
			break
		w *= math.exp( math.log(numpy.random.random()) / size )

	return output
