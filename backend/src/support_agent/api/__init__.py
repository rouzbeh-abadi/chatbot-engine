"""HTTP: routes, request models, and SSE framing.

The engine emits `Event` objects and has no opinion about transport -- turning
them into an HTTP response is this layer's job.
"""
