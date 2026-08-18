"""The engine's HTTP surface.

Thin by design: parse, authenticate, delegate to `services/`, serialise. Logic
that appears here is logic that cannot be reused by a non-HTTP caller.
"""
