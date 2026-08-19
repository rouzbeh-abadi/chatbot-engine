"""The application's own database: business data the AI engine never touches.

Bookings, flights and support tickets live here because they are this
application's domain, and because the tools that read them must run with the
calling user's permissions.
"""
