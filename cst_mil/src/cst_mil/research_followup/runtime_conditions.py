"""Shared host-capacity observations for local research measurements."""
from datetime import datetime, timezone

def conditions() -> dict:
    """Observed host capacity; unrelated system load is recorded, not controlled."""
    import psutil

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return dict(
        at=datetime.now(timezone.utc).isoformat(),
        logical_cpus=psutil.cpu_count(logical=True),
        physical_cpus=psutil.cpu_count(logical=False),
        memory_total_bytes=memory.total,
        memory_available_bytes=memory.available,
        memory_used_percent=memory.percent,
        swap_total_bytes=swap.total,
        swap_used_bytes=swap.used,
        swap_used_percent=swap.percent,
        system_load_control="unrelated processes are not stopped or controlled",
    )
