import pytest
import threading
import time
from backend.ring_buffer import RingBuffer

def test_ring_buffer_initialization():
    rb = RingBuffer(capacity=10)
    assert rb.capacity() == 10
    assert rb.size() == 0
    assert rb.is_empty() is True
    assert rb.is_full() is False

    with pytest.raises(ValueError):
        RingBuffer(capacity=0)

def test_ring_buffer_append_and_latest():
    rb = RingBuffer(capacity=5)
    
    sample1 = {"ts": 1, "val": 10}
    sample2 = {"ts": 2, "val": 20}
    
    rb.append(sample1)
    assert rb.latest() == sample1
    assert rb.size() == 1
    
    rb.append(sample2)
    assert rb.latest() == sample2
    assert rb.size() == 2

def test_ring_buffer_fifo_overwrite():
    rb = RingBuffer(capacity=3)
    
    for i in range(5):
        rb.append({"val": i})
        
    assert rb.is_full() is True
    assert rb.size() == 3
    # Buffer should contain 2, 3, 4
    assert rb.get(0) == {"val": 2}
    assert rb.get(1) == {"val": 3}
    assert rb.get(2) == {"val": 4}
    assert rb.latest() == {"val": 4}

def test_ring_buffer_append_batch():
    rb = RingBuffer(capacity=4)
    rb.append_batch([{"val": i} for i in range(6)])
    
    # Should only keep the last 4 elements: 2, 3, 4, 5
    assert rb.size() == 4
    assert rb.get(0) == {"val": 2}
    assert rb.get(3) == {"val": 5}
    assert rb.latest() == {"val": 5}

def test_ring_buffer_window():
    rb = RingBuffer(capacity=5)
    for i in range(10):
        rb.append({"val": i})
        
    # Elements in buffer: 5, 6, 7, 8, 9
    win = rb.window(3)
    assert len(win) == 3
    assert win == [{"val": 7}, {"val": 8}, {"val": 9}]
    
    # If we ask for more than size, it caps at size
    win_all = rb.window(10)
    assert len(win_all) == 5

def test_ring_buffer_clear():
    rb = RingBuffer(capacity=5)
    rb.append({"val": 1})
    rb.clear()
    assert rb.is_empty() is True
    assert rb.size() == 0
    assert rb.latest() is None

def test_ring_buffer_stress():
    # Insert 100,000 items and ensure it performs well and overwrites correctly
    rb = RingBuffer(capacity=1000)
    for i in range(100000):
        rb.append({"val": i})
        
    assert rb.size() == 1000
    assert rb.latest() == {"val": 99999}
    assert rb.get(0) == {"val": 99000}

def test_ring_buffer_thread_safety():
    rb = RingBuffer(capacity=1000)
    
    def worker(start_val, count):
        for i in range(count):
            rb.append({"thread_val": start_val + i})
            
    threads = []
    for t_idx in range(4):
        t = threading.Thread(target=worker, args=(t_idx * 10000, 500))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Total items inserted = 2000, capacity = 1000
    # The size should exactly be 1000, no weird race conditions on size
    assert rb.size() == 1000
    assert rb.is_full() is True
