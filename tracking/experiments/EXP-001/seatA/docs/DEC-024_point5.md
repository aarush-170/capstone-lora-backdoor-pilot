5. **Overall verdict wording (clarification, made before any result exists; no threshold value changes).**
   The thresholds text defines Partial as "none Evades, at least one Partial". That leaves one case uncovered: an adapter is rated Evades, but no lambda has both seeds rated Evades. Read literally, that case falls into "Not feasible", which would understate it. The rule is read as:
   - **Feasible:** at least one lambda where both seeds are rated Evades.
   - **Partial:** no lambda has both seeds rated Evades, but at least one adapter is rated Evades or Partial.
   - **Not feasible at this setting:** no adapter is rated Evades or Partial.
   The per-adapter counts (Evades, Partial, Partial (utility cost), Fails) are reported in full next to the overall label.
   Raised by seat A on Sat 24 Oct 2026. Needs confirmation by seats B and C before the verdict script is run; a disagreement is logged as a deviation before anyone sees results.
