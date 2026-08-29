# Mermaid diagnostic (temporary — will be deleted)

## 0. Version info

```mermaid
info
```

## 1. Trivial flowchart

```mermaid
flowchart TB
    A --> B
```

## 2. Subgraph only

```mermaid
flowchart TB
    subgraph Group1
        A["node a"]
        B["node b"]
    end
    A --> B
```

## 3. Cylinder shape only

```mermaid
flowchart TB
    A["normal node"]
    B[("cylinder node")]
    A --> B
```

## 4. classDef/class only

```mermaid
flowchart TB
    A["node a"]
    B["node b"]
    classDef proc fill:#e0e7ff,stroke:#3730a3,color:#0f172a;
    class A,B proc;
```

## 5. Trivial sequence diagram

```mermaid
sequenceDiagram
    participant A
    participant B
    A->>B: hello
    B-->>A: hi
```
