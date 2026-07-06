from __future__ import annotations


def check_snapshot_analysis(*, workdir, agent_output: str, model: str, test_case_name: str) -> dict:
    del workdir, model, test_case_name
    output = agent_output.casefold()
    required_groups = {
        "optimizer allocation": ("optimizer.py", "optimizer state", "optimizer"),
        "activation allocation": ("activation.py", "activation"),
        "inactive cached memory": ("inactive", "cached", "reserved"),
        "deallocation timing": ("free_requested", "free completed", "pending free", "dealloc"),
        "leak signal": ("leak", "retained", "reference cycle", "callback"),
        "multi-step pattern": ("step", "5", "five", "repeated"),
    }
    missing = [
        label
        for label, choices in required_groups.items()
        if not any(choice in output for choice in choices)
    ]
    return {
        "passed": not missing,
        "details": "missing: " + ", ".join(missing) if missing else "all expected signals present",
    }


def check_leak_diff_analysis(
    *, workdir, agent_output: str, model: str, test_case_name: str
) -> dict:
    del workdir, model, test_case_name
    output = agent_output.casefold()
    required_groups = {
        "growing site named": ("metrics.py", "log_step_loss"),
        "growth quantified": ("180", "20.00 mib", "200.00 mib", "4 mib per step", "45 block"),
        "stable retention excluded": ("optimizer", "parameter"),
        "not the leak": ("expected", "stable", "not the leak", "not leaking", "unchanged"),
        "fix": ("detach", ".item()", "cpu"),
    }
    missing = [
        label
        for label, choices in required_groups.items()
        if not any(choice in output for choice in choices)
    ]
    return {
        "passed": not missing,
        "details": "missing: " + ", ".join(missing) if missing else "all expected signals present",
    }


def check_oom_analysis(*, workdir, agent_output: str, model: str, test_case_name: str) -> dict:
    del workdir, model, test_case_name
    output = agent_output.casefold()
    required_groups = {
        "oom event": ("oom", "out of memory"),
        "failed request size": ("2.00 gib", "2 gib", "2147483648", "requested"),
        "device free": ("device_free", "device free", "64.00 mib", "64 mib"),
        "dominant live memory": ("optimizer", "parameter", "activation"),
        "pending free": ("active_awaiting_free", "pending free", "awaiting free"),
        "inactive cached context": ("inactive", "cached", "reserved", "empty_cache"),
        "actionable mitigation": ("batch", "checkpoint", "detach", "optimizer", "loss"),
    }
    missing = [
        label
        for label, choices in required_groups.items()
        if not any(choice in output for choice in choices)
    ]
    return {
        "passed": not missing,
        "details": "missing: " + ", ".join(missing)
        if missing
        else "all expected OOM signals present",
    }
