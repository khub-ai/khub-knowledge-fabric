# Trial 20260420T021147Z — ls20 L1 INITIAL_ASSESSMENT diff

- TUTOR: `claude-sonnet-4-6`  latency=37954 ms
- PUPIL: `google/gemma-4-26b-a4b-it`  latency=42232 ms

## Parse status

- TUTOR: OK
- PUPIL: OK

## Section summaries

### elements
| model | summary |
|---|---|
| TUTOR | 14 (agent:2, collectible:2, counter:2, readout:2, unknown:3, wall:3) |
| PUPIL | 7 (agent:2, readout:1, target:3, wall:1) |

### similar_groups
| model | summary |
|---|---|
| TUTOR | 3 groups |
| PUPIL | 2 groups |

### initial_strategy
| model | summary |
|---|---|
| TUTOR | first_action='ACTION1', goal=Navigate the agent (white cross) through the maze-like floor |
| PUPIL | first_action=None, goal=Navigate the agent through the maze to reach the target area |

### probes
| model | summary |
|---|---|
| TUTOR | 5 probes |
| PUPIL | 2 probes |

## Comparison metrics

- **element_count_tutor**: 14
- **element_count_pupil**: 7
- **element_bbox_iou_avg**: 0.0
- **element_function_agreement_rate**: None
- **group_count_tutor**: 3
- **group_count_pupil**: 2
- **similar_group_member_overlap**: 0.0
- **strategy_first_action_match**: False
- **strategy_first_action_tutor**: ACTION1
- **strategy_first_action_pupil**: None
- **strategy_primary_goal_len_ratio**: 0.425
- **strategy_open_questions_count_tutor**: 3
- **strategy_open_questions_count_pupil**: 3
- **probe_count_tutor**: 5
- **probe_count_pupil**: 2
- **probe_valid_rate_tutor**: 1.0
- **probe_valid_rate_pupil**: 0.5

## Probe execution

### TUTOR
- P1: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0
- P2: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0
- P3: OK → REGION_DELTA=20, SCORE_DELTA=0, REGION_DELTA=6
- P4: OK → STATE=NOT_FINISHED, ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0
- P5: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0

### PUPIL
- P1: REJECTED (["observation 'REGION_DELTA [124,505,241,638]': REGION_DELTA bbox out of bounds for 64x64: 'REGION_DELTA [124,505,241,638]'"])
- P2: OK → ELEMENT_MOVED=moved=None post=None, ELEMENT_MOVED=moved=None post=None
