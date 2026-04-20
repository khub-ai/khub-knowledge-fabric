# Trial 20260420T021421Z — ls20 L1 INITIAL_ASSESSMENT diff

- TUTOR: `claude-sonnet-4-6`  latency=34791 ms
- PUPIL: `google/gemma-4-26b-a4b-it`  latency=67837 ms

## Parse status

- TUTOR: OK
- PUPIL: OK

## Section summaries

### elements
| model | summary |
|---|---|
| TUTOR | 13 (agent:2, collectible:2, decor:1, hazard:1, portal:1, readout:3, target:1, wall:2) |
| PUPIL | 7 (agent:2, decor:1, readout:1, target:2, wall:1) |

### similar_groups
| model | summary |
|---|---|
| TUTOR | 3 groups |
| PUPIL | 1 groups |

### initial_strategy
| model | summary |
|---|---|
| TUTOR | first_action='ACTION3', goal=Navigate the agent through the maze-like floor to reach the  |
| PUPIL | first_action=None, goal=Navigate the agent through the maze to reach the target area |

### probes
| model | summary |
|---|---|
| TUTOR | 5 probes |
| PUPIL | 2 probes |

## Comparison metrics

- **element_count_tutor**: 13
- **element_count_pupil**: 7
- **element_bbox_iou_avg**: 0.0
- **element_function_agreement_rate**: None
- **group_count_tutor**: 3
- **group_count_pupil**: 1
- **similar_group_member_overlap**: 0.0
- **strategy_first_action_match**: False
- **strategy_first_action_tutor**: ACTION3
- **strategy_first_action_pupil**: None
- **strategy_primary_goal_len_ratio**: 0.574
- **strategy_open_questions_count_tutor**: 3
- **strategy_open_questions_count_pupil**: 3
- **probe_count_tutor**: 5
- **probe_count_pupil**: 2
- **probe_valid_rate_tutor**: 1.0
- **probe_valid_rate_pupil**: 0.5

## Probe execution

### TUTOR
- P1: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=50
- P2: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0
- P3: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=0
- P4: OK → SCORE_DELTA=0, STATE=NOT_FINISHED, REGION_DELTA=25
- P5: OK → ELEMENT_MOVED=moved=True post=[8, 13, 62, 53], REGION_DELTA=52, STATE=NOT_FINISHED

### PUPIL
- P1: OK → ELEMENT_MOVED=moved=None post=None, REGION_DELTA=52
- P2: REJECTED (["observation 'REGION_DELTA [955,205,985,850]': REGION_DELTA bbox out of bounds for 64x64: 'REGION_DELTA [955,205,985,850]'"])
