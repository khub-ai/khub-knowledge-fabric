Level ls20-9607627b: 0/7 levels completed across two full sessions (62+ turns total). Core mechanics remain poorly understood. Here is everything confirmed and suspected:

ACTIONS: ACTION1 moves the agent/cursor upward by ~5 rows (confirmed multiple times). ACTION3 moves it left by ~5 cols (confirmed). ACTION2 and ACTION4 effects remain ambiguous — they may move the cursor down/right respectively, or they may be state-mutation triggers. No visible state change on the playing field was ever observed from any action except cursor movement and progress bar decrement.

ELEMENTS: The colored block (colors 12/9, starting ~rows 44-46, cols 34-38) is the movable agent. The white cross (~rows 30-32, cols 20-22) is the change_indicator — its role is still unclear; it may be a separate cursor or a fixed marker. The top bordered box (rows 8-15, cols 32-40, color-3 border) and bottom-left bordered box (rows 53-62, cols 1-10) are reference glyphs encoding the target pattern. The yellow bar (bottom strip) is a MOVE BUDGET that decrements ~1-2 per action — it does NOT fill toward a win. Starting budget ~42 actions. Red dots bottom-right are remaining lives.

WIN CONDITION: Unknown. Hypothesis: navigate the agent to specific cells matching the reference glyph pattern, which triggers progress. Alternative hypothesis: ACTION2 or ACTION4 stamps/imprints a color at the agent's current position, and the player must imprint the correct pattern matching the reference glyph.

CRITICAL TRAPS: Both sessions were wasted on probing. With only ~42 actions total, any more than 3-4 test moves is fatal. Do not probe — commit to a strategy immediately.

RECOMMENDED NEXT ATTEMPT: Press ACTION2 once, ACTION4 once to confirm directions. Then navigate agent to the reference glyph coordinates (rows 11-13, cols 35-37) and try ACTION2/ACTION4 there to see if it stamps color. If no change, try overlapping agent with the white cross.