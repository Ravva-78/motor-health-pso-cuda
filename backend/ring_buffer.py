import threading
import numpy as np
from backend.logger import get_logger

logger = get_logger(__name__)

class RingBuffer:
    """
    High-performance thread-safe Ring Buffer backed by a NumPy object array.
    Designed for O(1) inserts and window reads. Overwrites old telemetry when full.
    """
    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        if self._capacity <= 0:
            logger.error("Attempted to initialize RingBuffer with capacity <= 0")
            raise ValueError("Capacity must be > 0")
            
        # Pre-allocate numpy array for O(1) memory footprint
        self._buffer = np.empty(self._capacity, dtype=object)
        
        self._head = 0  # Where to write the next element
        self._size = 0  # How many elements are currently in the buffer
        self._lock = threading.Lock()
        self._has_overflowed = False
        
        logger.info(f"RingBuffer initialized with capacity {self._capacity}")
        
    def initialize(self):
        """Explicitly clear and prepare the buffer."""
        self.clear()
        
    def append(self, sample):
        """Append a single sample to the buffer in O(1) time."""
        with self._lock:
            if self._size == self._capacity and not self._has_overflowed:
                logger.warning("RingBuffer overflow - overwriting old telemetry data")
                self._has_overflowed = True
                
            self._buffer[self._head] = sample
            self._head = (self._head + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1

    def append_batch(self, samples):
        """Append multiple samples to the buffer efficiently."""
        with self._lock:
            n = len(samples)
            if n == 0:
                return
                
            # If batch is larger than capacity, only take the last 'capacity' elements
            if n > self._capacity:
                samples = samples[-self._capacity:]
                n = self._capacity
                
            if self._size + n > self._capacity and not self._has_overflowed:
                logger.warning("RingBuffer overflow during batch append - overwriting old data")
                self._has_overflowed = True
                
            # Calculate remaining space until the end of the array
            space_to_end = self._capacity - self._head
            
            if n <= space_to_end:
                self._buffer[self._head:self._head + n] = samples
            else:
                self._buffer[self._head:] = samples[:space_to_end]
                self._buffer[:n - space_to_end] = samples[space_to_end:]
                
            self._head = (self._head + n) % self._capacity
            self._size = min(self._capacity, self._size + n)

    def latest(self):
        """Retrieve the single most recent element in O(1) time."""
        with self._lock:
            if self._size == 0:
                return None
            idx = (self._head - 1) % self._capacity
            return self._buffer[idx]

    def get(self, index):
        """Get an item from the buffer where 0 is the oldest available element."""
        with self._lock:
            if index < 0 or index >= self._size:
                raise IndexError(f"Index {index} out of bounds for size {self._size}")
            
            if self._size < self._capacity:
                actual_idx = index
            else:
                actual_idx = (self._head + index) % self._capacity
            return self._buffer[actual_idx]

    def window(self, size):
        """Get the most recent 'size' elements as a list."""
        with self._lock:
            if size <= 0:
                return []
            size = min(size, self._size)
            if size == 0:
                return []
            
            # calculate start index
            start_idx = (self._head - size) % self._capacity
            
            if start_idx < self._head:
                # no wrapping needed
                return self._buffer[start_idx:self._head].tolist()
            else:
                # wraps around the end of the array
                return np.concatenate((self._buffer[start_idx:], self._buffer[:self._head])).tolist()

    def clear(self):
        """Clear all data from the buffer."""
        with self._lock:
            self._buffer.fill(None)
            self._head = 0
            self._size = 0
            self._has_overflowed = False
            logger.info("RingBuffer cleared")

    def is_empty(self):
        """Return True if the buffer is empty."""
        with self._lock:
            return self._size == 0

    def is_full(self):
        """Return True if the buffer has reached maximum capacity."""
        with self._lock:
            return self._size == self._capacity

    def size(self):
        """Return the current number of elements in the buffer."""
        with self._lock:
            return self._size

    def capacity(self):
        """Return the maximum capacity of the buffer."""
        return self._capacity
