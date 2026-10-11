---
id: {{id}}
title: {{title}}
owner: {{owner}}
rq: {{rq}}
status: planned
branch:
created: {{date}}
updated: {{date}}
---
# {{id}}: {{title}}

## Question
(One sentence: what are we trying to find out?)

## Hypothesis
(What we expect, and what would change our mind. Hypotheses are not results.)

## Design
(Model, data, settings, what is held fixed, what varies, how many adapters or runs, who runs it where.)

## Thresholds
Written: (date time zone) BEFORE any run. Agreed by: (names, date).
(Pass or fail numbers. Fixed once agreed; any change afterwards goes under Deviations with the reason.)

## Approaches
One row per approach tried, including the ones that failed. Keep dead ends: the reason is the point.
Status is one of: planned, running, tried, failed, chosen, abandoned.

| id | what differs | result | status | why |
|---|---|---|---|---|
| A | | | planned | |

## Results
(Link every number to a log row or file: `logs/...`, `tracking/runs/RUN-....json`, a commit.)

## Deviations
(Anything different from the design or thresholds, with the date and the reason.)

## Verdict
(Pass or fail against the thresholds, in plain words, and what it does NOT show.)

## Next
