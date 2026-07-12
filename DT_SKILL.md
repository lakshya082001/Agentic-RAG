# DT Generation Skill

## Trigger
Use this skill when asked to generate a clinical Decision Tree (DT) for PA. Inputs will always include: HCPCS code, payer, LOB, and raw criteria text.

---

## Output Format

```
DECISION TREE: Eligibility Criteria for {CODE} {PAYER} {LOB}
==============================================================
[root] ROOT: Eligibility Criteria for {CODE} {PAYER} {LOB} (AND|OR)
│  ├─ [q001] Question?
│  │
│  ├─ [q002] Question?
│  │
│  └─ [q003] Last question?
│

==============================================================
```

