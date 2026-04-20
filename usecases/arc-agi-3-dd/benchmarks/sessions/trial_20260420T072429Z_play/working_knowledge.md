LESSONS_FROM_LAST_RUN (YOU wrote this at the end of your
previous play session.  It REPLACES any contradictory
claim below — trust it over the Round-2 assessment and
over any prior_knowledge):
Level ls20-9607627b (and likely ls-family siblings): 45 turns spent, 0/7 levels completed. Key findings and corrections to prior knowledge:

The white cross (change_indicator, ~rows 30-32, cols 20-22) IS movable. ACTION1 moves it upward by ~5 rows. ACTION3 moves it left by ~5 cols. ACTION2 and ACTION4 likely move it down and right respectively (not confirmed but strongly suspected by symmetry). The cross is NOT a fixed marker — it is a cursor that the player repositions over target cells.

The four actions are directional movement keys for the change_indicator cursor, not state-mutation actions per the prior knowledge. The cursor must be navigated to specific cells, then presumably an action fires a mutation at that location. However, it remains unclear which action (if any) actually triggers the mutation vs. just moves the cursor — this was never isolated.

The yellow progress bar DECREMENTS with each action taken (starts ~42, loses 1-2 per action). It does NOT fill up toward a win — it is a move budget/timer. Running it to zero causes level reset and costs one red attempt. Do NOT spend turns on blind probing; the budget is tight (~20-25 effective moves).

The reference glyphs (top-center bordered box, bottom-left bordered box) encode the target pattern. The internal color-9 arrangement in the top glyph is what must be recreated on the playing field.

Critical trap this run: Spent all 45 turns probing action effects instead of acting. The move budget is ~42 actions total — probing more than 5-6 actions is fatal.

Recommended strategy: Use ACTION1/ACTION3 to move the change_indicator cursor to cells matching the reference glyph pattern, then use ACTION2 or ACTION4 to stamp/imprint the target color. Test ACTION2 and ACTION4 immediately (1-2 presses each) to confirm they move the cursor down/right, then navigate directly to target cells.

---

ELEMENTS (from your Round-2 revised assessment):
  #1 reference_glyph_top bbox=[8, 32, 15, 40] fn=target — Small bordered box (color 3 border, color 5 interior) with internal color-9 pattern. Per prior knowledge, this is a reference glyph encoding the target state the player must recreate. Never moved in any probe. The internal pattern (rows 11-13, cols 35-37) shows a specific arrangement of color-9 cells.
  #2 change_indicator bbox=[30, 20, 32, 22] fn=switch — White cross (color 0 and 1 pixels) in the main floor area. Per prior knowledge this is a change indicator marking where the next state transition will occur when an action fires. Never moved in any probe.
  #4 main_floor_area bbox=[24, 14, 50, 53] fn=wall — Large color-3 region forming the navigable playing field. Contains the agent, change indicator, and colored blocks. Has gaps (color-4 regions) within it.
  #6 agent_cursor bbox=[44, 34, 46, 38] fn=agent — The movable colored block (colors 12 and 9) that responds to ACTION1 (moves up 5 cells) and ACTION3 (moves left 5 cells). This is the player-controlled cursor/agent. Starting position approximately rows 44-46, cols 34-38 in the initial frame.
  #7 reference_glyph_bottom_left bbox=[53, 1, 62, 10] fn=target — Small bordered box in bottom-left area with internal L-shaped glyph. Per prior knowledge, this is the second reference glyph in a matched pair with element 1. Encodes a target pattern.
  #8 progress_bar bbox=[61, 13, 62, 54] fn=readout — Yellow (color 11) horizontal bar in the bottom UI strip. Acts as a progress counter that decrements by 1 per action taken (observed: 42->41 per action, 42->37 over 5 actions). Win condition likely requires filling or matching a target value.
  #9 remaining_attempts bbox=[62, 56, 63, 63] fn=counter — Red (color 8) dots/squares at far right of bottom strip. Per prior knowledge these are remaining attempts/lives. Did not change during probes (no failed episodes yet).
  #10 left_sidebar bbox=[0, 0, 51, 3] fn=decor — Consistent column of color-5 tiles on the left edge. Decorative border or UI frame. Never changed in any probe.
  #11 bottom_ui_strip bbox=[60, 12, 63, 63] fn=readout — Bottom strip containing the yellow progress bar and red attempt indicators. The yellow fill decrements with each action (counter_changes confirmed). This is the main HUD area.
  #15 gap_in_floor bbox=[29, 29, 37, 33] fn=hazard — Rectangular gap (color-4 background) within the main color-3 floor area. Could be a pit/hazard that the agent must avoid, or a passage to another section.
  #16 colored_blocks_in_floor bbox=[44, 34, 50, 38] fn=unknown — The color-12 (blue) and color-9 (orange) cells in the main floor area at rows 44-49. Some may be part of the agent's current position, others may be fixed colored tiles that form a pattern to match.

PRIMARY_GOAL: Move the agent cursor (element 6, colors 12/9) to recreate the reference glyph pattern shown in element 1 (top bordered box) on the playing field, tracked by the yellow progress bar
STRATEGY_NOTES: ACTION1 moves the agent upward (dr=-5 per press). The agent needs to navigate toward the reference glyph area or toward the change indicator. The progress bar decrements with each action so efficiency matters. Need to determine ACTION2 and ACTION4 directions, then navigate the agent to match the reference pattern. The reference glyph (element 1) shows a specific color-9 arrangement that must be recreated.
OPEN_QUESTIONS:
  - What direction does ACTION2 move the agent? (P2 showed no movement - possibly blocked or ACTION2 is not a movement action)
  - What direction does ACTION4 move the agent?
  - What is the exact target pattern to recreate - is it the glyph in element 1 or element 7?
  - Does the progress bar fill UP toward a goal or does it represent remaining moves (depleting)?
  - What happens when the agent reaches the change indicator (white cross) - does it trigger a state change?

PRIOR_KNOWLEDGE (operator-injected, treat as given):
Prior knowledge from previous plays of the ls* game family (ls20, ls21, ...).

Treat these as strong priors — learned from other levels and other ls-series
games — but still subject to per-level evidence from the grounding pack.

1. The white cross (color 0 pixel cluster in the main floor area) is NOT
   the agent.  In ls-family games it is a "change indicator": it marks the
   cell where the next state transition will happen when an action fires.

2. ACTION1..ACTION4 are NOT directional movement keys.  They are
   "state-mutation" actions: each one attempts to change a property of the
   cell under the cross (or a neighbouring cell) — for example, toggling a
   floor tile, activating a switch, or imprinting a glyph.

3. The yellow horizontal bar at the bottom of the frame is a PROGRESS
   counter.  The level win condition is to fill it completely, usually by
   making the playing-field state match a reference pattern encoded
   somewhere on the frame.

4. The three red squares at the bottom-right are REMAINING ATTEMPTS (lives).
   Each failed episode reduces them by one.

5. Small bordered boxes with an internal glyph (here: the top-centre
   bordered room and the bottom-left corner icon) are USUALLY
   "reference glyphs" — they encode the target state the player must
   recreate on the playing field.  They often appear in matched PAIRS:
   one serves as the reference, the other as a write-target.

6. A win in ls-family games typically requires recreating the reference
   glyph (item 5) on the playing field using the state-mutation actions
   (item 2), tracked by the progress bar (item 3).

These priors come from other levels — they may still be wrong for this
specific level.  If the grounding pack directly contradicts any of them,
trust the grounding pack.