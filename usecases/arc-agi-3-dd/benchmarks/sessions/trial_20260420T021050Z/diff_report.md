# Trial 20260420T021050Z — ls20 L1 INITIAL_ASSESSMENT diff

- TUTOR: `claude-sonnet-4-6`  latency=35400 ms
- PUPIL: `??`  latency=? ms

## Parse status

- TUTOR: OK
- PUPIL: FAIL — URLError: <urlopen error [Errno 11001] getaddrinfo failed>

## Section summaries

### elements
| model | summary |
|---|---|
| TUTOR | 14 (agent:3, collectible:2, counter:1, readout:1, unknown:4, wall:3) |
| PUPIL | — |

### similar_groups
| model | summary |
|---|---|
| TUTOR | 3 groups |
| PUPIL | — |

### initial_strategy
| model | summary |
|---|---|
| TUTOR | first_action='ACTION3', goal=Navigate the agent (white cross) through the maze-like floor |
| PUPIL | — |

### probes
| model | summary |
|---|---|
| TUTOR | 5 probes |
| PUPIL | — |

## Comparison metrics


## Probe execution

### TUTOR
- P1: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=50
- P2: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0
- P3: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=50
- P4: OK → SCORE_DELTA=0, STATE=NOT_FINISHED, REGION_DELTA=0
- P5: OK → STATE=NOT_FINISHED, ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0

### PUPIL
