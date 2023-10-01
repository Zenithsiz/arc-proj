"""
Utilities
"""

from typing import TypeVar, Mapping

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
