#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ACTIVE_STATES = {"active_allocated", "active_awaiting_free"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Path to .pickle/.pkl or .json snapshot")
    parser.add_argument("--top", type=int, default=10, help="Number of top sites to print")
    parser.add_argument(
        "--diff",
        type=Path,
        default=None,
        metavar="LATER_SNAPSHOT",
        help="Second snapshot from a later point in the same run; reports live-memory "
        "growth per allocation site instead of a single-snapshot summary",
    )
    parser.add_argument(
        "--max-trace-events",
        type=int,
        default=1_000_000,
        help="Cap on device_trace events analyzed; older events beyond the cap are "
        "dropped and the truncation is reported",
    )
    args = parser.parse_args()

    snapshot = load_snapshot(args.snapshot)
    if args.diff is not None:
        later = load_snapshot(args.diff)
        diff = diff_snapshots(snapshot, later, top_n=args.top)
        print_diff_markdown(args.snapshot, args.diff, diff)
        return 0
    analysis = analyze_snapshot(snapshot, top_n=args.top, max_trace_events=args.max_trace_events)
    print_markdown(args.snapshot, analysis)
    return 0


def load_snapshot(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    with path.open("rb") as handle:
        return pickle.load(handle)


def analyze_snapshot(
    snapshot: Any, *, top_n: int, max_trace_events: int = 1_000_000
) -> dict[str, Any]:
    segments = list(iter_segments(snapshot))
    blocks = [block for segment in segments for block in segment.get("blocks", [])]
    traces = list(iter_trace_entries(snapshot))
    truncated_events = max(0, len(traces) - max_trace_events)
    if truncated_events:
        traces = traces[-max_trace_events:]

    total_reserved = sum(int(segment.get("total_size", 0)) for segment in segments)
    by_state: dict[str, int] = defaultdict(int)
    requested_by_state: dict[str, int] = defaultdict(int)
    site_bytes: dict[str, int] = defaultdict(int)
    site_blocks: Counter[str] = Counter()
    top_blocks = []

    for block in blocks:
        state = str(block.get("state", "unknown"))
        size = int(block.get("size", 0))
        requested = int(block.get("requested_size", size))
        by_state[state] += size
        requested_by_state[state] += requested
        if state in ACTIVE_STATES:
            site = frame_label(block.get("frames", []))
            site_bytes[site] += requested
            site_blocks[site] += 1
            top_blocks.append(
                {
                    "state": state,
                    "size": size,
                    "requested_size": requested,
                    "site": site,
                }
            )

    top_sites = sorted(site_bytes.items(), key=lambda item: item[1], reverse=True)[:top_n]
    top_blocks.sort(key=lambda item: item["requested_size"], reverse=True)
    trace_summary = summarize_traces(traces)
    fragmentation = fragmentation_summary(segments)
    flags = issue_flags(
        total_reserved=total_reserved,
        by_state=by_state,
        site_blocks=site_blocks,
        trace_summary=trace_summary,
        fragmentation=fragmentation,
    )

    return {
        "segments": len(segments),
        "blocks": len(blocks),
        "truncated_trace_events": truncated_events,
        "total_reserved": total_reserved,
        "by_state": dict(sorted(by_state.items())),
        "requested_by_state": dict(sorted(requested_by_state.items())),
        "top_sites": [
            {"site": site, "bytes": size, "blocks": site_blocks[site]} for site, size in top_sites
        ],
        "top_blocks": top_blocks[:top_n],
        "trace_summary": trace_summary,
        "fragmentation": fragmentation,
        "issue_flags": flags,
    }


def live_bytes_by_site(snapshot: Any) -> tuple[dict[str, int], Counter[str]]:
    site_bytes: dict[str, int] = defaultdict(int)
    site_blocks: Counter[str] = Counter()
    for segment in iter_segments(snapshot):
        for block in segment.get("blocks", []):
            if str(block.get("state", "")) in ACTIVE_STATES:
                site = frame_label(block.get("frames", []))
                site_bytes[site] += int(block.get("requested_size", block.get("size", 0)))
                site_blocks[site] += 1
    return dict(site_bytes), site_blocks


def diff_snapshots(earlier: Any, later: Any, *, top_n: int) -> dict[str, Any]:
    earlier_bytes, earlier_blocks = live_bytes_by_site(earlier)
    later_bytes, later_blocks = live_bytes_by_site(later)

    deltas = []
    for site in set(earlier_bytes) | set(later_bytes):
        delta = later_bytes.get(site, 0) - earlier_bytes.get(site, 0)
        if delta == 0:
            continue
        deltas.append(
            {
                "site": site,
                "delta_bytes": delta,
                "earlier_bytes": earlier_bytes.get(site, 0),
                "later_bytes": later_bytes.get(site, 0),
                "earlier_blocks": earlier_blocks.get(site, 0),
                "later_blocks": later_blocks.get(site, 0),
            }
        )
    deltas.sort(key=lambda item: item["delta_bytes"], reverse=True)

    earlier_total = sum(earlier_bytes.values())
    later_total = sum(later_bytes.values())
    return {
        "earlier_live_bytes": earlier_total,
        "later_live_bytes": later_total,
        "delta_live_bytes": later_total - earlier_total,
        "growing_sites": [d for d in deltas if d["delta_bytes"] > 0][:top_n],
        "shrinking_sites": [d for d in deltas if d["delta_bytes"] < 0][-top_n:],
    }


def fragmentation_summary(segments: list[dict[str, Any]]) -> dict[str, Any]:
    # inactive space split across multiple blocks (within or across segments)
    # is unusable for any single request larger than the biggest piece
    largest_inactive = 0
    total_inactive = 0
    inactive_blocks = 0
    for segment in segments:
        for block in segment.get("blocks", []):
            if str(block.get("state", "")) == "inactive":
                size = int(block.get("size", 0))
                total_inactive += size
                inactive_blocks += 1
                largest_inactive = max(largest_inactive, size)
    return {
        "total_inactive": total_inactive,
        "largest_inactive_block": largest_inactive,
        "inactive_blocks": inactive_blocks,
    }


def iter_segments(snapshot: Any) -> list[dict[str, Any]]:
    segments = snapshot.get("segments", []) if isinstance(snapshot, dict) else snapshot
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def iter_trace_entries(snapshot: Any) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    traces = snapshot.get("device_traces", [])
    if not isinstance(traces, list):
        return []
    entries: list[dict[str, Any]] = []
    for trace in traces:
        if isinstance(trace, list):
            entries.extend(entry for entry in trace if isinstance(entry, dict))
    return entries


def frame_label(frames: Any) -> str:
    if not isinstance(frames, list) or not frames:
        return "(no stack)"
    normalized = [frame for frame in frames if isinstance(frame, dict)]
    if not normalized:
        return "(no stack)"
    for frame in normalized:
        filename = str(frame.get("filename", ""))
        if "/site-packages/torch/" not in filename and "/torch/" not in filename:
            return format_frame(frame)
    return format_frame(normalized[0])


def format_frame(frame: dict[str, Any]) -> str:
    filename = frame.get("filename", "<unknown>")
    line = frame.get("line", "?")
    name = frame.get("name", "<unknown>")
    return f"{filename}:{line} in {name}"


def summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(str(entry.get("action", "unknown")) for entry in traces)
    alloc_event_counts: Counter[str] = Counter()
    alloc_event_bytes: Counter[str] = Counter()
    live_bytes = 0
    peak_live_bytes = 0
    requested_at: dict[int, tuple[int, int]] = {}
    pending_free_addrs: set[int] = set()
    completed_free_latencies = []
    oom_events = []

    for index, entry in enumerate(traces):
        action = str(entry.get("action", "unknown"))
        size = int(entry.get("size", 0) or 0)
        address = entry.get("addr", entry.get("address"))
        key = int(address) if isinstance(address, int) else None
        time_value = trace_time(entry, index)

        if action == "alloc":
            live_bytes += size
            peak_live_bytes = max(peak_live_bytes, live_bytes)
            site = frame_label(entry.get("frames", []))
            alloc_event_counts[site] += 1
            alloc_event_bytes[site] += size
        elif action == "free_requested" and key is not None:
            requested_at[key] = (time_value, size)
            pending_free_addrs.add(key)
        elif action == "free_completed":
            live_bytes = max(0, live_bytes - size)
            if key is not None:
                pending_free_addrs.discard(key)
                if key in requested_at:
                    start, requested_size = requested_at.pop(key)
                    completed_free_latencies.append(
                        {
                            "address": key,
                            "size": size or requested_size,
                            "delay": time_value - start,
                        }
                    )
        elif action == "oom":
            oom_events.append(
                {
                    "index": index,
                    "size": size,
                    "device_free": entry.get("device_free"),
                    "site": frame_label(entry.get("frames", [])),
                }
            )

    top_alloc_events = [
        {
            "site": site,
            "events": count,
            "bytes": alloc_event_bytes[site],
        }
        for site, count in alloc_event_counts.most_common(10)
    ]

    return {
        "events": len(traces),
        "actions": dict(sorted(action_counts.items())),
        "estimated_peak_live_bytes": peak_live_bytes,
        "pending_free_count": len(pending_free_addrs),
        "completed_free_latencies": completed_free_latencies[:10],
        "top_allocation_events": top_alloc_events,
        "oom_events": oom_events,
    }


def trace_time(entry: dict[str, Any], fallback: int) -> int:
    for key in ("time_us", "timestamp", "sequence_nr", "sequence_number"):
        value = entry.get(key)
        if isinstance(value, int):
            return value
    return fallback


def issue_flags(
    *,
    total_reserved: int,
    by_state: dict[str, int],
    site_blocks: Counter[str],
    trace_summary: dict[str, Any],
    fragmentation: dict[str, Any] | None = None,
) -> list[str]:
    flags = []
    inactive = by_state.get("inactive", 0)
    active_allocated = by_state.get("active_allocated", 0)
    awaiting = by_state.get("active_awaiting_free", 0)
    if total_reserved and inactive / total_reserved >= 0.25:
        flags.append("reserved memory includes a large inactive cached portion")
    if active_allocated and inactive / max(active_allocated, 1) >= 0.25:
        flags.append("reserved-vs-live gap may indicate caching or fragmentation")
    if fragmentation and fragmentation["inactive_blocks"] >= 2:
        largest = fragmentation["largest_inactive_block"]
        total = fragmentation["total_inactive"]
        if total and largest < total:
            flags.append(
                "inactive memory is split across multiple blocks "
                f"(largest reusable piece {format_bytes(largest)} of "
                f"{format_bytes(total)} inactive); consider "
                "PYTORCH_ALLOC_CONF=expandable_segments:True"
            )
    if awaiting:
        flags.append("some memory is active_awaiting_free; check stream ordering and record_stream")
    if trace_summary.get("pending_free_count", 0):
        flags.append("trace has free_requested events without matching free_completed events")
    if trace_summary.get("oom_events"):
        flags.append("snapshot contains OOM events")
    repeated_sites = [site for site, count in site_blocks.items() if count >= 3]
    if repeated_sites:
        flags.append("repeated live blocks from the same stack may indicate retained tensors")
    return flags


def print_markdown(path: Path, analysis: dict[str, Any]) -> None:
    live = analysis["by_state"].get("active_allocated", 0)
    awaiting = analysis["by_state"].get("active_awaiting_free", 0)
    inactive = analysis["by_state"].get("inactive", 0)
    reserved = analysis["total_reserved"]

    print(f"# Memory Snapshot Summary: {path}")
    print()
    print("## Totals")
    print(f"- Segments: {analysis['segments']}")
    print(f"- Blocks: {analysis['blocks']}")
    print(f"- Reserved: {format_bytes(reserved)}")
    print(f"- Active allocated: {format_bytes(live)}")
    print(f"- Active awaiting free: {format_bytes(awaiting)}")
    print(f"- Inactive cached: {format_bytes(inactive)}")
    fragmentation = analysis.get("fragmentation", {})
    if fragmentation.get("total_inactive"):
        print(
            f"- Largest single reusable inactive block: "
            f"{format_bytes(fragmentation['largest_inactive_block'])}"
        )
    if analysis.get("truncated_trace_events"):
        print(
            f"- NOTE: {analysis['truncated_trace_events']} oldest trace events were "
            "dropped (over --max-trace-events); totals from traces are partial"
        )
    print()

    print("## Top Live Allocation Sites")
    if analysis["top_sites"]:
        for item in analysis["top_sites"]:
            pct = 100 * item["bytes"] / max(live + awaiting, 1)
            print(
                f"- {format_bytes(item['bytes'])} ({pct:.1f}%, {item['blocks']} block(s)): "
                f"{item['site']}"
            )
    else:
        print("- No active allocation stack traces found.")
    print()

    traces = analysis["trace_summary"]
    if traces["top_allocation_events"]:
        print("## Trace Allocation Sites")
        for item in traces["top_allocation_events"]:
            print(
                f"- {item['events']} alloc event(s), {format_bytes(item['bytes'])}: {item['site']}"
            )
        print()

    print("## Deallocation And Trace Signals")
    print(f"- Trace events: {traces['events']}")
    print(f"- Actions: {traces['actions']}")
    peak_live = format_bytes(traces["estimated_peak_live_bytes"])
    print(f"- Estimated peak live bytes from trace: {peak_live}")
    print(f"- Pending frees: {traces['pending_free_count']}")
    if traces["completed_free_latencies"]:
        print("- Completed free delays:")
        for item in traces["completed_free_latencies"]:
            size = format_bytes(item["size"])
            print(f"  - addr={item['address']} size={size} delay={item['delay']}")
    if traces["oom_events"]:
        print("- OOM events:")
        for item in traces["oom_events"]:
            device_free = item.get("device_free")
            device_free_text = format_bytes(device_free) if isinstance(device_free, int) else "?"
            print(
                f"  - index={item['index']} requested={format_bytes(item['size'])} "
                f"device_free={device_free_text} site={item['site']}"
            )
    print()

    print("## Issue Flags")
    if analysis["issue_flags"]:
        for flag in analysis["issue_flags"]:
            print(f"- {flag}")
    else:
        print("- No obvious allocator issue flags from this snapshot summary.")


def print_diff_markdown(earlier_path: Path, later_path: Path, diff: dict[str, Any]) -> None:
    print(f"# Memory Snapshot Diff: {earlier_path} -> {later_path}")
    print()
    print("## Live Memory")
    print(f"- Earlier: {format_bytes(diff['earlier_live_bytes'])}")
    print(f"- Later: {format_bytes(diff['later_live_bytes'])}")
    print(f"- Delta: {format_bytes(diff['delta_live_bytes'])}")
    print()
    print("## Growing Allocation Sites (leak candidates)")
    if diff["growing_sites"]:
        for item in diff["growing_sites"]:
            print(
                f"- +{format_bytes(item['delta_bytes'])} "
                f"({format_bytes(item['earlier_bytes'])} -> {format_bytes(item['later_bytes'])}, "
                f"{item['earlier_blocks']} -> {item['later_blocks']} block(s)): {item['site']}"
            )
    else:
        print("- No allocation site grew between the two snapshots.")
    print()
    print("## Shrinking Allocation Sites")
    if diff["shrinking_sites"]:
        for item in diff["shrinking_sites"]:
            print(
                f"- {format_bytes(item['delta_bytes'])} "
                f"({format_bytes(item['earlier_bytes'])} -> {format_bytes(item['later_bytes'])}): "
                f"{item['site']}"
            )
    else:
        print("- No allocation site shrank between the two snapshots.")


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} B"


if __name__ == "__main__":
    raise SystemExit(main())
