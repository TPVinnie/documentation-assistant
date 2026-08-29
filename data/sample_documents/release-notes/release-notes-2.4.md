---
title: Release Notes
category: release-note
version: 2.4
effective_date: 2024-07-15
status: current
supersedes: release-notes-2.3.md
---

# Release 2.4

This release increases the default request timeout from 30 seconds to 60 seconds to accommodate
larger exports.

# Known Issues

Clients using SDK version 1.2 or earlier may continue to observe a 30 second timeout due to a
client-side caching bug that has not yet been patched.
