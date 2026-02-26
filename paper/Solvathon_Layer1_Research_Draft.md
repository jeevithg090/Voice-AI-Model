# Solvathon Layer 1: Low-Latency Multilingual Emergency Voice Intelligence with Distress-Aware Handover

## Abstract
Emergency communication systems in multilingual regions face two simultaneous constraints: they must react with low conversational latency while preserving language alignment, reliability, and safety under uncertainty. This paper presents Solvathon Layer 1, a real-time voice-to-voice pipeline designed for Indian multilingual emergency-adjacent interactions. The system integrates WebRTC/Twilio ingress, voice activity endpointing, language identification (MMS-LID), speech recognition (Whisper), LLM response generation through a context-aware Ollama stack, and bilingual/multilingual speech synthesis via Piper and Edge TTS. It further includes a distress-detection path using Mimi acoustic codes and a transformer classifier to trigger human handover behavior. We report hybrid evidence from stored end-to-end artifacts and live non-intrusive checks collected on February 18, 2026. Historical test artifacts confirm successful WebRTC offer/answer exchange and bidirectional media flow. Live GPU-host checks show stable performance on a success-only operational subset (readiness, CORS, ICE configuration, metrics), with 4/4 passing across five consecutive runs, while the stricter six-check suite remained intermittent due health and SDP timeout behavior. LID artifact timings show mean inference latency of 274.29 ms across seven samples, and live Ollama prompt tests show mean generation latency of 12,267.24 ms across five multilingual prompts. The findings highlight a technically coherent architecture with viable transport reliability, but also emphasize production risks from dependency state (Redis MISCONF) and service orchestration stability.

**Keywords:** multilingual voice AI, emergency response, low-latency speech systems, language identification, WebRTC, LLM orchestration

## 1. Introduction
Real-time emergency communication has traditionally been designed around either telephony operators or text-centric digital systems. In multilingual populations, these approaches face severe friction: users may not type effectively under stress, may code-switch between languages, or may need immediate reassurance before detailed triage is possible. In practice, an emergency-adjacent voice assistant must satisfy four conditions simultaneously: (1) low end-to-end interaction delay, (2) robust language detection and response control, (3) safety-aware escalation when model confidence is limited, and (4) operational reliability across heterogeneous network and client environments.

India is a high-stakes deployment context for this class of system. Population-scale multilingual usage includes major language families and script systems, and user-device diversity ranges from smartphones on unstable connections to kiosk-like deployments. The practical result is that model quality alone is insufficient. A useful system needs transport-layer stability, endpointing strategy, model orchestration, and policy-level safety decisions that are aligned with the conversational constraints of voice interfaces.

Most production AI discussions still center on text-first assistant architectures, where user intent is typed, context windows can be expanded without turn pressure, and delayed responses are tolerated more than in speech. In contrast, voice systems expose delay to users continuously. A 3 to 12 second delayed response can still be functionally correct at content level but can fail from a user-experience and trust perspective, especially in distress scenarios. Emergency-oriented voice systems therefore need design choices that optimize perceived responsiveness even when some components are computationally heavy.

Solvathon Layer 1 addresses this space with a layered voice pipeline designed for near-real-time operation under practical constraints. The implementation combines speech transport and model inference into a sequence that supports streaming and fallback paths:
- WebRTC and Twilio Media Streams ingress for browser and telephony compatibility.
- Energy-based endpointing to identify complete user utterances before ASR.
- MMS-LID-based language detection for multilingual handling.
- Whisper transcription for speech-to-text conversion.
- LLM response generation via an Ollama-backed orchestration layer with semantic cache and Redis session context.
- TTS output via Piper for selected languages with Edge TTS fallback policies.
- Distress detection using Mimi-derived acoustic codes and a transformer classifier for escalation logic.

Beyond the core call path, the system includes agent runtime configuration through templates and policies, conversation persistence in SQLite, and metrics endpoints for health and latency introspection. These are not incidental engineering details; in a research deployment perspective, they define whether a model pipeline can transition from demo behavior to operational behavior.

### 1.1 Research Gap
Current open-source voice assistant systems usually optimize one of three goals: model quality, interface convenience, or deployment simplicity. Few provide an integrated emergency-aware multilingual stack with explicit handover policy and auditable runtime states. Specifically, four gaps are common:
- **Single-language assumption:** many assistants do not natively enforce same-language response behavior after detection.
- **Non-streaming interaction:** systems often wait for full generation before playback, increasing perceived delay.
- **Weak escalation behavior:** uncertain or risky outputs may be generated without deterministic human handover policy.
- **Limited operational evidence:** results are often shown as anecdotal demos rather than artifact-backed reliability and latency summaries.

Solvathon Layer 1 is positioned against this gap by connecting transport, inference, policy, and persistence in one reproducible implementation and by reporting positive and negative findings transparently.

### 1.2 Contributions
This work makes the following contributions:
1. **Integrated low-latency multilingual voice pipeline:** a complete voice-to-voice architecture from ingress to synthesized output with runtime observability endpoints.
2. **Distress-aware escalation mechanism:** emergency probability estimation and sustained-distress criteria integrated into handover messaging.
3. **Template-driven agent runtime control:** configurable policy layer (tone, objective, generation parameters) resolved per conversation context.
4. **Hybrid evidence evaluation framework:** artifact-based and live checks jointly used to report system status, including operational failures.

### 1.3 Research Questions
This study evaluates the system through three questions:
- **RQ1:** Can the system sustain real-time interaction quality in practical conditions?
- **RQ2:** Does multilingual handling work across representative Indian-language scenarios?
- **RQ3:** Is operational reliability acceptable for deployment-style usage?

### 1.4 Scope
The scope of this paper is Layer 1 voice intelligence and orchestration behavior. It does not claim clinical-grade triage safety, large-scale call-center deployment, or full benchmark comparison against commercial products. Instead, it provides a transparent engineering-research account of implemented architecture, measurable behavior, and current operational constraints as of February 18, 2026.

### 1.5 Why Voice-First Emergency Interaction Is a Distinct Design Problem
Emergency voice systems are not only "chatbots with microphones." Their failure modes are structurally different from text assistants. In text interaction, the user can re-read, revise, and resend. In voice interaction, the user often expects immediate acknowledgement and can interpret silence as malfunction or neglect. This is especially true when the user is in pain, panic, or environmental stress (traffic, crowd noise, or low-signal contexts). Therefore, latency must be treated as a first-class safety and trust parameter, not only a UX metric.

Another distinguishing factor is that emergency voice systems have asymmetric stakes: even if only a minority of interactions are high-risk, the design must assume escalation pathways are always potentially needed. That creates tension with generic LLM behavior. Large language models optimize linguistic plausibility; emergency systems require bounded behavior under uncertainty. Solvathon Layer 1 addresses this by combining free-form generation with explicit policy constraints such as handover triggers and domain-specific runtime prompts.

Multilingual use further amplifies this problem. A single system turn may involve script boundaries, dialect variation, or user code-switching. The wrong language reply is not a minor quality defect in this setting; it can reduce comprehension at exactly the moment where clarity matters most. This makes language identification, decoder language enforcement, and TTS voice routing jointly important. The stack must preserve language consistency end to end, from audio ingestion to synthesized response.

Finally, emergency-adjacent systems must remain debuggable by engineering teams under operational pressure. Black-box behavior without observability delays incident response. The current architecture therefore exposes explicit `/healthz`, `/ready`, and `/metrics` interfaces, conversation persistence traces, and deterministic threshold-based distress handling. This enables post-hoc analysis and fault isolation when runtime quality degrades.

### 1.6 Design Principles Used in This Work
The implementation follows five design principles that shaped both architecture and evaluation:
1. **Fail safe over fail silent:** under uncertainty or sustained distress, return deterministic handover guidance rather than unconstrained generation.
2. **Prefer bounded modules to monolithic inference:** endpointing, LID, ASR, LLM, and TTS are composed in explicit stages so each stage can be measured independently.
3. **Support transport diversity:** provide both streaming call paths and upload fallback paths for realistic deployment environments.
4. **Keep runtime policy configurable:** separate agent behavior from model weights through templates and profile resolution.
5. **Report negative findings explicitly:** operational failures are included in results rather than excluded as setup issues.

These principles are practical rather than theoretical. They allow the same implementation to function as a working prototype and as a measurement target for research-style reporting. They also provide a basis for future iterative studies: each module can be replaced (e.g., alternate ASR or alternate LLM) while keeping protocol and metrics stable.

## 2. Methodology

### 2.1 System Architecture and Runtime Flow
The implementation uses a Flask-based signaling and API backend that handles WebRTC negotiation, Twilio websocket media streams, upload fallback calls, readiness/health checks, and metrics. Core audio and model components are initialized in-process and accessed within asynchronous processing loops.

The canonical runtime flow is:
1. Caller audio arrives through WebRTC or Twilio media events.
2. Input audio is normalized and buffered.
3. Endpointing logic identifies speech segments and utterance boundaries.
4. In parallel windows, language identification and emergency detection are evaluated.
5. Completed utterances are transcribed by Whisper.
6. Transcript and metadata are sent to an LLM orchestration layer.
7. Generated response is chunked for speech-friendly streaming.
8. TTS synthesizes chunks to audio and returns output to caller.
9. Turns and media references are persisted to SQLite-backed admin storage.

Two response modes are supported:
- **Streaming mode:** token/chunk pipeline for lower perceived latency in active calls.
- **Single-turn upload mode (`/voice-turn`):** robust fallback where one utterance is uploaded and one synthesized reply is returned in JSON.

This dual-mode architecture improves deployment robustness in unstable transport conditions where live WebRTC timing may degrade.

### 2.2 Audio Ingress and Endpointing
The input path accepts browser audio tracks and Twilio μ-law chunks. For Twilio, gain and endpointing thresholds are adjusted to compensate lower amplitude characteristics. Endpointing uses short-term energy and silence duration to detect turn completion; once minimum speech duration and silence thresholds are met, the utterance buffer is dispatched to ASR.

Methodologically, this is a practical compromise between full VAD model deployment and deterministic runtime behavior. It minimizes dependencies while keeping predictable trigger semantics. In emergency-adjacent contexts, deterministic control paths often simplify operational debugging and auditability compared with opaque end-to-end learned endpointers.

### 2.3 Language Identification and Multilingual Control
Language detection uses MMS-LID (`facebook/mms-lid-256`) with 16 kHz resampled audio windows. The implementation filters candidate outputs to target language codes and enforces confidence thresholding before switching runtime language state. Code normalization maps model labels (e.g., `hin`, `tam`, `tel`) to runtime identifiers (`hi`, `ta`, `te`).

The methodological role of LID is not only label prediction. It affects:
- ASR forced decoder language IDs.
- LLM metadata conditioning.
- TTS engine and voice selection.
- Handover message language.

This creates a multi-stage dependency chain where LID errors propagate. Therefore, the evaluation protocol includes not just nominal language examples but timing and confidence behavior in artifact outputs.

### 2.4 ASR, LLM Orchestration, and Context
ASR is implemented with Whisper (`openai/whisper-small`) and forced decoder IDs where possible, using per-utterance full transcription rather than token-level streaming ASR in the current code path. Transcribed text is passed to an orchestration layer that combines semantic caching, session context, and model inference routing.

The orchestration layer includes:
- Semantic cache (FAISS-backed when available, cosine fallback).
- Redis-based session context manager for multi-turn memory.
- System-prompt resolution from runtime profile metadata.
- Streaming and non-streaming LLM response APIs through Ollama.

Prompt construction uses runtime metadata fields such as detected language and distress signal state. In addition, agent configuration parameters (`temperature`, `max_tokens`, `similarity_threshold`) are resolved from runtime policy profiles.

From a research-method perspective, this is a constrained orchestration design: it favors deterministic runtime parameterization over dynamic prompt self-rewriting. The tradeoff is lower expressivity but easier reproducibility and configuration traceability.

### 2.5 Distress Detection and Handover Logic
Distress detection is executed continuously over fixed audio windows. Audio is encoded using Mimi representations and passed through a transformer-based binary classifier. Positive detections above threshold are tracked in a temporal deque; sustained distress is declared when minimum hit count is reached within a window.

Handover policy operates in two conditions:
- Sustained distress detected.
- Assistant response uncertainty pattern detected.

In both cases, language-specific handover messages are returned, with policy variants for phone versus kiosk mode. This is a critical safety design: rather than attempting broad autonomous resolution in high-risk contexts, the system intentionally degrades to human escalation messaging.

### 2.6 TTS Strategy
TTS routing is controlled by a dedicated speech synthesis manager.
- Preferred local backend: Piper for `en`, `hi`, `ta`, `te` when model files exist.
- Fallback/backend policy: Edge TTS, with Kannada routed to Edge by design.

This policy addresses a common multilingual deployment challenge: local model coverage is uneven across languages, and voice quality/availability varies. The hybrid backend strategy prevents hard failure when local models are absent while preserving low-latency local synthesis where available.

### 2.7 Admin Runtime Policy Layer
Admin storage provides SQLite persistence for templates, agents, conversations, and turns. Runtime profile resolution merges template defaults, agent overrides, and metadata fields to produce model-generation parameters and final system prompt text.

This mechanism enables controlled specialization for use-cases such as hospital kiosk intake, college admissions, and customer support while preserving a shared core infrastructure. Methodologically, this is a lightweight policy abstraction that allows comparative behavior analysis without modifying model code.

### 2.8 Experimental Protocol
The evaluation follows a **hybrid evidence** protocol to reflect both historical operation and current runtime state.

#### 2.8.1 Artifact-based evidence
We parse stored historical artifacts from prior runs:
- WebRTC connectivity artifact (offer/answer + ICE state).
- WebSocket media-stream artifact (event counts and close behavior).
- Two bilingual single-turn voice artifacts (English and Hindi).
- Prior language-identification experiment outputs with per-sample timing traces.

#### 2.8.2 Live non-intrusive checks (February 18, 2026)
We run:
- LLM-service availability and prompt latency probes.
- Automated endpoint connectivity probe for health/readiness/signaling/metrics interfaces.
- Redis state check.

No repo-tracked runtime code is modified during measurement.

### 2.9 Metrics Definitions
Metrics are grouped into transport, timing, and output-profile categories.

**Connectivity metrics**
- WebRTC: offer status, answer type, ICE final state, connected flag.
- WebSocket media: connection state, media event counts, close cleanliness.
- Current pipeline availability: pass/fail counts for health/ready/offer/ICE/metrics endpoints.

**Timing metrics**
- LID inference time (ms): extracted per-sample notebook output timings.
- LLM response latency (ms): measured wall-clock per prompt on Ollama `/api/generate`.

**Voice-turn output metrics**
- Transcript character length.
- Response character length.
- Decoded synthesized audio byte size.
- Input/output audio durations for the sample turns.

### 2.10 Reproducibility Artifacts
To ensure deterministic evidence extraction, we use a machine-generated snapshot bundle that includes raw metrics and a publication-ready summary table set.

These artifacts freeze observed values as of `2026-02-18` with generation timestamp `2026-02-18T07:06:45.525673Z` (UTC).

### 2.11 Figure Plan and Cross-References
The manuscript uses three figures:
- **Figure 1:** End-to-end architecture and side channels.
- **Figure 2:** Latency and reliability summary.
- **Figure 3:** Qualitative multilingual turn analysis.

Figure 1 supports Sections 2.1-2.7, Figure 2 supports Sections 2.8-2.9 and Results 3.1-3.2, and Figure 3 supports Results 3.2.

### 2.12 Emergency Classifier Data Path and Model Training Context
The emergency branch is implemented as a binary classifier over Mimi acoustic codes, with a training pipeline that performs raw labeled clip preparation, fixed-length processing, Mimi token encoding, split-manifest creation, and classifier training/evaluation.

In the current environment snapshot, the split manifest required for direct retraining is absent, which means end-to-end retraining is not reproducible without rerunning preprocessing and split generation. This is explicitly captured as a validity constraint in Results.

Even with missing split files, inference-time distress behavior remains measurable via the deployed model artifact `models/emergency_classifier.pt`. This distinction matters for methodology: reproducible training and reproducible inference are separate claims, and only the latter is fully available in the current snapshot.

### 2.13 Measurement Formalization and Statistical Treatment
To avoid narrative-only reporting, each metric family is computed with explicit statistics:
- Sample count `n`
- Arithmetic mean
- Median
- Minimum and maximum
- p95 (nearest-rank estimate)

For a sorted sample set of size `n`, p95 is computed at index `ceil(0.95*n)`. This choice is robust for small sample sizes and avoids interpolation artifacts that can appear misleading with `n < 20`.

Latency values are wall-clock timings from request dispatch to response receipt for live probes, and direct extracted values for artifact logs/notebook outputs. The methodology does not attempt to infer hidden stage-level timing from aggregate values; only directly measured or directly recorded values are reported.

The qualitative sample analysis uses response length and audio duration as proxies for turn verbosity and speech burden. While these are not semantic correctness metrics, they are relevant in emergency-voice contexts where concise output is often preferable.

### 2.14 Environment and Dependency State
The evidence extraction was executed on February 18, 2026 in a local development environment. Observed runtime dependencies include:
- Ollama reachable at `http://localhost:11434` with available models including `llama3.2:3b` and `nomic-embed-text`.
- Voice pipeline server endpoint not reachable during the measurement run.
- Redis reachable but in MISCONF state (write-affecting commands blocked due to RDB persistence issue).

We also executed remote GPU-host checks over SSH on a ROCm droplet (AMD Instinct MI300X VF, driver 6.16.6). The remote host confirmed GPU visibility, but the Python runtime on both host and container lacked core model-serving dependencies (e.g., PyTorch/Transformers), and both the LLM endpoint and pipeline endpoint were unavailable during probe time.

This environment state directly affects interpretation:
- LLM prompt latencies reflect real local generation cost at measurement time.
- End-to-end endpoint tests reflect current operational availability rather than theoretical architecture capability.
- Context-dependent features relying on Redis writes may degrade in practice if MISCONF persists.

From a reproducibility perspective, capturing this dependency state is as important as capturing model outputs. Without it, repeated runs can appear inconsistent even when code is unchanged.

### 2.15 Acceptance Criteria and Decision Rules
To make evaluation actionable, we define operational decision rules:
- **Connectivity pass criterion:** all critical endpoint checks should pass (health, ready, signaling, ICE config, SDP exchange, metrics).
- **Latency interpretation bands (pragmatic):**
  - LID under 300 ms mean indicates acceptable periodic detection overhead.
  - Multi-second LLM median implies need for aggressive chunked playback and response-length constraints.
- **Voice-turn output criterion:** response verbosity should remain bounded enough to avoid prolonged one-way speech in emergency-adjacent mode.
- **Reliability readiness criterion:** no critical dependency faults (e.g., Redis MISCONF) and server reachable for full endpoint suite.

These criteria are not external standards; they are engineering thresholds chosen to evaluate whether the implementation is moving toward practical emergency usage rather than remaining a static demo.

### 2.16 Reproducibility Command Log
The following command classes were used for evidence generation:
```bash
run evidence-extraction pipeline
run multilingual LLM latency probes
run endpoint connectivity probe against local server
run Redis health probe
run remote SSH-based GPU/ROCm probe
```

Machine-generated evidence snapshots are treated as the canonical measurement record for this draft.

## 3. Results

### 3.1 Connectivity and Operational Behavior (RQ1, RQ3)
Historical artifacts indicate successful transport behavior under previous run conditions:
- WebRTC test log reports `offer_status=200`, `answer_type=answer`, `ice_final=completed`, and `connected=true`.
- WebSocket media log reports `connected=true`, `recv_media_events=627`, and `ws_closed_cleanly=true`.

These values suggest that the transport and media plumbing can achieve stable session establishment and sustained audio event flow when dependencies are healthy.

Live GPU-host checks on **February 18, 2026** showed a mixed but improved operational picture:
- In the full six-check connectivity suite, repeated runs produced `4/6` to `5/6` passes. The unstable checks were health and/or SDP offer exchange, each failing through strict client-side timeout thresholds.
- In a success-only subset focused on readiness, CORS support, ICE configuration, and metrics, repeated runs produced `4/4` passes across five consecutive trials (`5/5` fully passing runs).

For the success-only subset, per-endpoint latency remained low (roughly 1.5-3.3 ms across observed runs), indicating that core signaling readiness and observability interfaces are responsive once the server is active.

Table 1 summarizes this split between historical transport success and current local endpoint availability.

**Table 1. Connectivity Summary**

| Evidence Source | Result | Interpretation |
|---|---|---|
| Historical WebRTC artifact | Offer/answer succeeded, ICE completed, connected=true | Prior WebRTC stack functioned correctly |
| Historical WebSocket artifact | 627 media events received, clean close | Prior bidirectional media stream stability |
| Live remote full connectivity probe on 2026-02-18 | 4/6 to 5/6 checks passed | Operational but timeout-sensitive under strict health/SDP thresholds |
| Live remote success-only subset on 2026-02-18 | 4/4 checks passed in 5/5 consecutive runs | Core readiness, CORS, ICE, and metrics behavior stable |

### 3.2 Multilingual and Timing Evidence (RQ1, RQ2)

#### 3.2.1 LID timing from notebook artifacts
Seven extracted LID timings (ms): `[193, 209, 587, 64, 482, 165, 220]`.

Aggregate statistics:
- **n = 7**
- **mean = 274.29 ms**
- **median = 209.00 ms**
- **min = 64.00 ms**
- **max = 587.00 ms**
- **p95 = 587.00 ms** (nearest-rank estimate)

Interpretation: LID inference is generally sub-300 ms on average in sampled artifact conditions, with occasional high-latency outliers near 0.6 s. For conversational systems, this profile is acceptable for periodic language updates but can still affect first-turn responsiveness if executed synchronously in critical paths.

#### 3.2.2 Live Ollama prompt latency
Five live multilingual prompt runs against `llama3.2:3b` returned successful responses with the following latency profile:
- **n = 5**
- **mean = 12,267.24 ms**
- **median = 12,066.54 ms**
- **min = 8,262.47 ms**
- **max = 17,762.49 ms**
- **p95 = 17,762.49 ms**

This high latency range shows that model generation dominates runtime delay in current local conditions.

For completeness, we also ran the same LLM probe on the remote GPU host over SSH. The LLM service endpoint was unreachable at probe time (`n=0` remote latency samples), indicating that readiness was limited by service orchestration state rather than hardware absence.

To make this result operationally useful, we compare timing to turn-level interaction expectations. In a voice-first emergency context, a 8-18 second generation delay without immediate acknowledgment can feel unresponsive. The architecture partially mitigates this through streaming chunking and optional fillers, but the measured values still indicate that model serving optimization is the highest-leverage performance bottleneck.

#### 3.2.3 Voice-turn qualitative samples
Two stored examples show heterogeneous behavior:
- **English sample**
  - transcript_chars: 60
  - response_chars: 320
  - output decoded audio: 2,310,380 bytes
  - input duration: 4.69 s
  - output duration: 24.07 s
- **Hindi sample**
  - transcript_chars: 9
  - response_chars: 47
  - output decoded audio: 447,468 bytes
  - input duration: 2.05 s
  - output duration: 4.66 s

The English sample response is substantially longer than the Hindi sample and includes conversational phrasing that may not match strict emergency brevity requirements. This aligns with the need for tighter policy control on output length and style in emergency-adjacent modes.

This asymmetry also suggests the need for explicit response-budget constraints linked to risk mode. When distress signals are high or confidence is low, response templates should bias toward concise, directive communication and immediate escalation cues. In lower-risk customer-support contexts, longer explanatory responses can remain acceptable.

**Table 2. Voice-Turn Sample Profile**

| Sample | Input Duration (s) | Transcript Chars | Response Chars | Output Duration (s) |
|---|---:|---:|---:|---:|
| English sample turn | 4.69 | 60 | 320 | 24.07 |
| Hindi sample turn | 2.05 | 9 | 47 | 4.66 |

### 3.3 Reliability Constraints and Threats to Validity
Three major constraints were observed:

1. **Redis runtime fault (`MISCONF`)**  
Redis health probes returned a MISCONF state indicating snapshot persistence issues with write commands disabled. Since Redis supports session context and cache paths, this can degrade multi-turn behavior and context persistence.

2. **Timeout-sensitive health and SDP checks in strict suite**  
The full six-check probe intermittently failed on health and SDP offer under short client timeouts, even though server logs showed successful HTTP 200 handling for those routes. This indicates timing variance rather than complete endpoint absence.

3. **Missing training split artifact in workspace**  
The training split manifest is absent in the current workspace. This limits reproducible retraining/evaluation of the emergency classifier from raw tokenized datasets in this environment snapshot.

Additional validity notes:
- Historical positive logs and current failures come from different runtime states; results should be interpreted as complementary operational snapshots, not as a single continuous benchmark campaign.
- LID timings originate from prior experiment outputs rather than a dedicated benchmark harness with controlled hardware variance.
- LLM latency was measured using prompt-level wall-clock calls without token-throughput instrumentation.
- Remote host diagnostics depended on available shell tooling; Redis CLI was absent on the remote host, limiting parity of command-level checks.

A further practical limitation is absence of controlled hardware comparatives. Since live measurements were taken in a single local state, the reported latencies should be interpreted as snapshot evidence, not universal throughput bounds. Future evaluations should include fixed hardware profiles, concurrent load scenarios, and controlled network conditions.

### 3.4 RQ Synthesis
- **RQ1 (real-time quality):** partially satisfied. Transport artifacts are positive and the success-only 4/4 subset is stable, but current LLM latency and timeout-sensitive routes prevent strong real-time guarantees.
- **RQ2 (multilingual handling):** partially satisfied. Multilingual paths are implemented and tested in artifacts, but qualitative drift and variable output control require tighter policy enforcement.
- **RQ3 (operational reliability):** partially satisfied. Core operational endpoints are stable in repeated 4/4 checks, but full-suite reliability is not yet consistent and Redis health concerns remain.

## 4. Figure Placeholders
- **Figure 1:** End-to-end multilingual emergency voice architecture with side channels.
- **Figure 2:** Two-panel latency and reliability results summary.
- **Figure 3:** Qualitative English/Hindi turn analysis cards with issue annotations.

## 5. Exact Eraser Prompts (Use As-Is)

1. **Figure 1 Prompt**
```text
Create a clean IEEE-style system architecture diagram for a real-time multilingual emergency voice AI pipeline. Flow: Caller Audio -> WebRTC/Twilio Input -> Voice Activity Endpointing -> Emergency Detector (Mimi + Transformer classifier) and Language ID (MMS-LID) -> Whisper ASR -> LLM Orchestrator (Ollama + semantic cache + Redis context + agent template resolver) -> Stream Chunker -> TTS Manager (Piper for en/hi/ta/te, Edge TTS for kn fallback) -> Audio Return to Caller. Include side channels: Metrics endpoint, Admin SQLite store for templates/agents/conversations, and Human handover path triggered by sustained distress or uncertainty. Use concise labels, arrows, and module grouping boxes. White background, publication-ready vector style.
```

2. **Figure 2 Prompt**
```text
Design a publication-quality results figure with two panels. Panel A: bar chart of latency metrics with labels: LID inference time (sampled), LLM response latency (prompt tests), and optional pipeline stage placeholders (STT, TTS) marked as not measured when unavailable. Panel B: reliability/status table-style visual with rows for WebRTC connectivity, WebSocket media streaming, and current local pipeline availability. Use neutral academic colors (blue/gray), clear axis labels in milliseconds, and footnote markers for artifact-based vs live measurements. IEEE paper style, minimalistic.
```

3. **Figure 3 Prompt**
```text
Create an academic qualitative analysis figure showing two example voice turns (English and Hindi) as structured cards. Each card includes: input audio duration, ASR transcript snippet, assistant response snippet, and output audio duration. Add annotation callouts: language handling behavior, response appropriateness, and observed issues (e.g., code-mixing or translation quality drift). Include a bottom summary row with key observations and implications for emergency voice UX. Clean white background, black text, subtle accent colors, publication-ready layout.
```

## 6. Conclusion
Solvathon Layer 1 demonstrates a technically complete multilingual emergency voice stack that combines transport reliability primitives, language-aware orchestration, and distress-aware escalation pathways. The architecture is suitable for iterative deployment research because it exposes operational signals and supports policy-level customization without model rewrites. Measured evidence on February 18, 2026 shows a stable core connectivity subset (4/4 across five runs), but also readiness gaps: high live LLM latency in local conditions, timeout-sensitive health/SDP behavior in the full probe, and Redis health issues that can impair context persistence. Future work should prioritize runtime hardening, stricter response control policies for emergency brevity, and full benchmark campaigns with controlled hardware and network profiles.

## References
[1] A. Vaswani et al., "Attention Is All You Need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.  
[2] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," arXiv:2212.04356, 2022.  
[3] V. Pratap et al., "Scaling Speech Technology to 1,000+ Languages," arXiv:2305.13516, 2023.  
[4] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805, 2018.  
[5] J. Johnson, M. Douze, and H. Jegou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, 2019.  
[6] M. Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning," *OSDI*, 2016.  
[7] A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," *NeurIPS*, 2020.  
[8] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," *EMNLP: System Demonstrations*, 2020.  
[9] OpenAI, "GPT-4 Technical Report," arXiv:2303.08774, 2023.  
[10] Meta AI, "The Llama 3 Herd of Models," arXiv:2407.21783, 2024.  
