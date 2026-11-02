# AI Tools and Disclosure Log

In accordance with hackathon rules and transparency guidelines, this document logs all AI tools, models, and assistants utilized during the design, development, and operation of **PatchPilot**.

| Tool / Model | Provider | Purpose / Phase | Notes |
|--------------|----------|-----------------|-------|
| **Antigravity (Agentic AI Assistant)** | Google DeepMind | Architecture design, scaffold generation, test development, documentation | Autonomous pair programming assistant in Antigravity IDE |
| **Gemini 2.5 Flash / Gemini 3.8 Flash** | Google (`google-genai` SDK) | Runtime LLM review layer: finding explanation, risk assessment, and automated minimal patch generation | Called at runtime via `patchpilot.llm.enrich` with JSON mode |

## Human Oversight & Code Integrity
- All AST rules, McCabe complexity calculations, and scoring ranking logic are deterministically implemented and verified with exhaustive automated unit tests.
- Gemini LLM generation is isolated as an enrichment layer, designed with graceful fallback so failure or lack of an API key never degrades core code review functionality.
