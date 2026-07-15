# Award-paper lessons and the conversion-experiment eval artifact (2026-07)

Seventeen award papers (ICLR/ICML 2026, ACL/EMNLP 2025) were read against our
situation after the interrogator conversion kill. During that review we also
audited our own conversion-run outputs. This doc records (1) a decisive eval
artifact found in our experiment, (2) the mechanisms and designs the papers
contribute, (3) the reshaped plan. Index of papers read:
`/Users/vein/Documents/research/paste-boundary-inference/papers/AWARD_PAPERS.md`
(cot-faithfulness-unlearning.pdf is corrupt; covered by reconstruction only).

## 1. The artifact: our reader prompt structurally forbids the required answers

`v2/replay_io.py` `SHARED_READER_TEMPLATE` instructs: "Return only the answer
span: a name, phrase, number, date, or list" and "If the evidence is
insufficient, answer exactly: Unknown" (max_tokens 128). BEAM's hard-ability
references require narrative/behavioral answers: event_ordering wants an
ordered multi-step narrative; contradiction_resolution wants
contradiction-flagging + a clarification request; multi_session wants an
evolution narrative. Measured abstention in the conversion run: Qwen-72B
answers literally "Unknown" on 60-69% of rows IN EVERY CONDITION INCLUDING THE
ORACLE CEILING; Llama-8B 21-31%, with many non-Unknown answers derailing into
continuations of the evidence text. The 72B reader was following instructions
correctly. This was foreshadowed by the never-diagnosed MiMo 57.1->35.2
shared-prompt collapse flagged in CONTEXT.md.

Scope of invalidation:
- The conversion experiment had NO DYNAMIC RANGE (all conditions floored at
  0-12%). Therefore BOTH headline conclusions are unsafe: the ceiling result
  ("acquisition is second-order") AND the interrogator-vs-adaptive-k null
  (nothing could have shown a difference at floor). The interrogator kill is
  downgraded from "clean kill" to "no evidence of conversion; re-measure
  under valid eval before final verdict."
- The offline reachability results (24.0% BEAM, LME decay-to-zero) are
  unaffected — no reader involved.
- The field-wide 0.00-0.08 on BEAM contradiction is still corroborating, but
  our own 0% adds nothing until re-measured.

## 2. What the award papers contribute

### Measurement discipline (apply before any mechanism claim)
- **Aptitude vs unreliability** (LLMs Get Lost in Multi-Turn, ICLR26): don't
  report single-sample means on multi-turn tasks; sample 3-5x per row, report
  90th percentile (aptitude) and 90-10 spread (unreliability). Multi-turn
  unreliability spreads reach ~50pp; a single-sample 0% is uninterpretable.
- **Judge validity audit** (multiple): (a) feed the GOLD answer through the
  judge — if it scores <2, the judge/format is broken; (b) manually review
  ~50 judge-rejected answers with relaxed criteria.
- **Task-appropriate prompt**: answer shape must be allowed by the prompt
  (narrative answers, no forced span, no hard abstention rule, max_tokens
  ~512 for list/narrative abilities).

### Mechanism candidates for hard-ability failure (distinguishable by cheap
interventions; naming the true one is the award-paper move — cf. Flexibility
Trap's "entropy degradation")
1. **Format suppression** (our artifact): prompt forbids the answer shape.
   Test: rerun with task-appropriate prompt.
2. **Premature commitment / lost-in-conversation** (ICLR26): readers lock in
   early pattern-matches on scattered evidence and cannot recover.
   Test: CONCAT control — same information, one well-structured block vs
   scattered turns. Gap = maximum recoverable by presentation. This bounds
   the chain-rendering bet BEFORE building it.
3. **Temporal-order degradation** (Flexibility Trap analog): readers cannot
   synthesize order from co-present unordered turns. Test: chronologically
   sorted vs scrambled injection.
4. **Prescriptive-norm override** (ACL25 response-sampling): readers resist
   statistically-unlikely-but-correct answers (the UPDATED value of a fact)
   in favor of canonical ones. Test: correlate knowledge_update errors with
   prior-consistency of the wrong answer.
5. **Difference-unawareness** (ACL25 fairness): alignment suppresses
   adjudicating between contradicting claims (refuse/average instead of
   discriminate). Test: two-step probe — "which claim is more recent?" then
   "answer"; if step 1 fails, compilation is irrelevant.
6. **Causal disuse of evidence** (CoT-faithfulness reconstruction): answers
   may be causally disconnected from context evidence. Test: corrupt key
   facts to plausible wrong values — invariant answers = evidence not in the
   causal path.
7. **Elasticity** (ACL25 resist-alignment): strong readers resist prompt-level
   perturbation proportionally to pretraining mass — a structural account of
   why every intervention lands at ~+0pp on strong readers; motivates
   rendering formats that ALIGN with pretraining distribution (dateline/CV
   "as of <date>" formats) rather than fight it.

### Paper-shape lessons
- Negative arcs win when they (a) name a mechanism measured at the failure
  point, (b) close escape hatches with falsified remediations, (c) build a
  positive method whose justification IS the mechanism (Flexibility Trap
  template).
- LingGym: structured DECLARATIVE rules (KP-type) beat raw data injection,
  and CoT adds nothing once structure is right — supersession chains are a
  KP-type intervention; test annotated chain vs raw chronological list, and
  expect the gain to shift the capability threshold (largest for weak
  readers) — frame accordingly.
- Source/sink decomposition (EMNLP25 filler-gap): map which abilities are
  "sources" (compilation transfers in) vs "sinks" (isolated); predictive
  features of sink-ness are a theory contribution.
- Benchmark-validity escape (infini-gram template): if the eval artifact
  generalizes — no evidence-complete system can score on BEAM's hard
  abilities under standard QA prompting because the references demand
  conversational BEHAVIOR — that is itself a main-conference finding about
  the benchmark, not a Findings note.
- Displacement risk (NSA): long-context progress erodes
  "compilation-for-tractability"; position on accuracy mechanism + query-time
  cost, and include a full-history long-context reader as a condition where
  feasible.

## 3. Reshaped plan (order matters; each step gates the next)

1. **Fix the eval** ($0): task-appropriate prompt per ability family
   (narrative allowed, no forced span/abstention, max_tokens 512); judge
   sanity check (gold-through-judge must score 2); keep everything else
   frozen.
2. **Re-run the four conversion conditions under valid eval** (~$10-15,
   3 samples/row for unreliability): this re-measures the true ceiling AND
   re-adjudicates the interrogator kill with dynamic range. Possible
   outcomes: ceiling opens -> acquisition matters again (interrogator may
   resurrect); ceiling stays shut with correct-format answers -> the
   reasoning-bound conclusion becomes SAFE and strong.
3. **Mechanism battery on the fixed eval** (~$10): CONCAT control,
   sorted-vs-scrambled, corruption test, two-step discrimination probe —
   one intervention per candidate mechanism above; name the winner.
4. **Chain rendering as the method** bet, now grounded: annotated
   supersession chain (KP-type) vs raw chronological list vs
   resolve-to-latest, across weak+strong readers, on knowledge_update +
   contradiction_resolution; framed as capability-threshold shift.
5. Paper spine: mechanism-named diagnosis of why evidence-complete readers
   fail conversational-memory reasoning + the rendering method built on it +
   the acquisition negative arc as motivation + benchmark-validity finding
   if step 2 confirms the artifact generalizes.
