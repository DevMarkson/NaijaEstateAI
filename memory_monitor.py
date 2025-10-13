"""Memory monitoring utilities for debugging memory issues."""

import gc
import os
import sys

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        # Fallback without psutil
        return {'error': 'psutil not available'}

def log_memory_usage(context=""):
    """Log current memory usage."""
    memory = get_memory_usage()
    if 'error' not in memory:
        print(f"Memory {context}: RSS={memory['rss_mb']:.1f}MB, "
              f"VMS={memory['vms_mb']:.1f}MB, "
              f"Percent={memory['percent']:.1f}%")
    else:
        print(f"Memory monitoring not available: {memory['error']}")

def force_cleanup():
    """Force garbage collection and memory cleanup."""
    collected = gc.collect()
    print(f"Garbage collector: collected {collected} objects")
    return collected

if __name__ == "__main__":
    log_memory_usage("startup")
    force_cleanup()
    log_memory_usage("after cleanup")