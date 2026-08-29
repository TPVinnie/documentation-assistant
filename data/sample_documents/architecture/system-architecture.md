---
title: System Architecture Overview
category: architecture
version: 1.4
effective_date: 2024-05-01
status: current
---

# Overview

The platform is composed of an API Gateway, an Auth Service, a primary Document Store, a Vector
Index, and a Worker Queue for asynchronous jobs.

# Data Flow

Client requests enter through the API Gateway, which authenticates the request against the Auth
Service before routing it to the appropriate backend service.

# Rate Limiting

The API Gateway enforces a global rate limit of 500 requests per hour per client, independent of
any endpoint-level limits configured on individual services.

# Failure Modes

If the Auth Service is unavailable, the API Gateway fails closed and rejects all requests that
require authentication, rather than allowing them through unauthenticated.
