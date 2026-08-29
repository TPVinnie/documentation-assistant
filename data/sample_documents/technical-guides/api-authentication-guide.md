---
title: API Authentication Guide
category: technical-guide
version: 2.1
effective_date: 2024-05-10
status: current
---

# Overview

The platform API supports both static API keys and OAuth2 bearer tokens for authentication.

# Obtaining an API Key

API keys are issued through the developer portal and are tied to a single project.

# Token Expiry

Access tokens expire after 24 hours. Refresh tokens are valid for 30 days from issuance.

# Rate Limiting

API requests are rate-limited to 1000 requests per hour per API key.

# Error Codes

A 401 response indicates invalid or missing credentials. A 429 response indicates the rate
limit has been exceeded.
