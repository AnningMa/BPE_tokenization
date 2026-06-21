import heapq


class LazyHeap:
    """
    Max-heap over (frequency, pair) with lazy deletion.

    Standard heapq doesn't support priority updates, so we use lazy deletion:
    when a pair's frequency changes, we push a new entry and mark the old one
    stale. Stale entries are skipped silently on pop.

    All frequencies are stored as positive ints in _freq; the heap stores
    negated values to simulate a max-heap with Python's min-heap.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, tuple[str, str]]] = []
        self._freq: dict[tuple[str, str], int] = {}

    def push(self, pair: tuple[str, str], freq: int) -> None:
        self._freq[pair] = freq
        heapq.heappush(self._heap, (-freq, pair))

    def update(self, pair: tuple[str, str], delta: int) -> None:
        if delta == 0:
            return
        new_freq = self._freq.get(pair, 0) + delta
        if new_freq <= 0:
            self._freq.pop(pair, None)
        else:
            self._freq[pair] = new_freq
            heapq.heappush(self._heap, (-new_freq, pair))

    def pop_best(self) -> tuple[tuple[str, str], int] | None:
        while self._heap:
            neg_freq, pair = heapq.heappop(self._heap)
            freq = -neg_freq
            # Valid entry: heap value matches current freq in _freq
            if self._freq.get(pair) == freq:
                del self._freq[pair]
                return pair, freq
            # Stale entry: skip and keep draining
        return None

    def get_freq(self, pair: tuple[str, str]) -> int:
        return self._freq.get(pair, 0)

    def __len__(self) -> int:
        return len(self._freq)
