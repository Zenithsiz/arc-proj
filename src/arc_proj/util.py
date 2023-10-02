"""
Utilities
"""

from typing import Mapping, TypeVar


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
	Formats time in seconds in the closest unit.
	"""
	assert s >= 0, "Seconds must be positive"

	hours   = s // 3600
	mins    = s // 60   - 60 * hours
	seconds = s // 1    - 60 * mins  - 3600 * hours
	milliseconds = 1e3 * (s - seconds)
	microseconds = 1e6 * (s - seconds)
	nanoseconds  = 1e9 * (s - seconds)

	if hours > 0:
		return f"{hours:.0f}h{mins:.0f}m{s%60:.2f}s"
	elif mins > 0:
		return f"{mins:.0f}m{s%60:.2f}s"
	elif seconds > 0:
		return f"{s:.2f}s"
	elif milliseconds >= 1.0:
		return f"{milliseconds:.2f}ms"
	elif microseconds >= 1.0:
		return f"{microseconds:.2f}μs"
	else:
		return f"{nanoseconds:.2f}ns"
