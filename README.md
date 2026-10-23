# 🛡️ PALADIN — Policy-Aware Layered Agentic Decision Intelligence

A composable governance framework for securing LLM-based agentic tool selection
through layered identity, attribute, and formal logic policies.

---

## Problem Statement

Modern autonomous AI agents operate in open, federated ecosystems using protocols like the
**Model Context Protocol (MCP)**. These agents select and invoke tools on behalf of users —
but current security models offer only **flat RBAC** ("can this agent use this tool?") or
**prompt-level guardrails** that are brittle and non-auditable.

This creates a critical **governance gap**: probabilistic LLM decisions must be grounded by
**non-repudiable identity**, **contextual risk awareness**, and **formal logic verification**
— all *before* any tool is executed.

### Research Questions

| # | Question |
|---|----------|
| **RQ1** | Does a composable governance stack (RBAC → ABAC → TRAC) provide measurably superior security over any single layer alone? |
| **RQ2** | Can a typed, ordered task-relational rule engine (TRAC) provide a deterministic safety floor for probabilistic LLM tool selection? |
| **RQ3** | Does PALADIN produce complete predicate traces sufficient for post-hoc audit of every tool-use decision? *(Evidence: trace logs in the Experiment Lab detail every predicate evaluation per task.)* |

---

## Technical Novelty

### Governance Layers Defined

| Layer | Full Name | What It Does |
|-------|-----------|--------------|
| **RBAC** | Role-Based Access Control | Grants or denies access based on the caller's assigned role. Answers: *"Is this agent's role allowed to use this tool?"* |
| **ABAC** | Attribute-Based Access Control | Evaluates contextual attributes (clearance, department, trust score, risk level, action type) to enforce fine-grained conditions beyond role membership. Answers: *"Given the current context, should this action be permitted?"* |
| **TRAC** | Task-Relational Access Control | A typed-fact-base rule engine with deterministic ordered evaluation that judges whether the selected tool bundle satisfies the task's capability requirements with correct domain alignment and task-tool relevance. Rules compose via derived predicates and fire in a policy-declared order with short-circuit denial. Operates post-inference, pre-execution. Answers: *"Is this the right set of tools for this task — and is the selection logically sound?"* |

### 1. Composable Layered Governance
Unlike flat RBAC systems, PALADIN enforces three complementary security layers
with distinct failure modes — each catching threats the others miss:

| Layer | Catches | Example |
|---|---|---|
| **RBAC** | Identity mismatches | Finance agent → DevOps tools |
| **ABAC** | Contextual risk | Right role but after-hours + high-risk write |
| **TRAC** | Logical inconsistency | Domain mismatch, capability gaps |

Ablation experiments prove each layer provides **irreplaceable value** —
removing any one leaves exploitable gaps.

### 2. TRAC: Task-Relational Access Control for Agentic Security
TRAC doesn't just ask *"Can this agent use this tool?"* — it asks:
> *"Does the selected tool bundle satisfy the mission's capability requirements
> with sufficient confidence and domain alignment?"*

- **Post-inference, pre-execution** verification gate
- **Advisory alerts** — TRAC can raise a non-blocking audit alert on an admitted
  bundle (e.g. a destructive write without a verifying read) without changing the
  ALLOW/DENY decision.
- **Complete predicate traces** — every decision is formally auditable

### 3. OPA Baseline Validation
PALADIN is validated against **Open Policy Agent (OPA)** — the CNCF-graduated
industry standard. Following OPA's Document Model, **policy logic lives in
generic Rego** while the RBAC allow-lists and ABAC attribute rules are loaded
as **`data` documents** — the very same YAML the Python engines read, with no
code generation. Real `opa eval` is then replayed against the Python engines:

- **Rule equivalence** — `opa eval` matches the Python RBAC / ABAC / TRAC
  decisions with **0 mismatches** (see `scripts/run_validation_suite.py`)
- **Single source of truth** — rules-as-data means OPA and Python evaluate the
  identical rule set; editing a rule (or Policy Studio) updates both at once
- **Layered advantage** — TRAC's deterministic capability / destructiveness
  predicates add an auditable safety-net on top of the flat allow/deny model
- **In-app proof** — Policy Studio's **OPA / Rego** tab shows the generic Rego +
  rules-as-data for every layer, runs a one-click **Verify parity** (`opa eval`
  vs the Python engine), and can start a live `opa run --server` to query the
  identical policies over OPA's REST API (`app/services/opa_runtime.py`)

### 4. Mission-Permission Decoupling
Unlike every existing RBAC/ABAC framework — which asks *"Is this caller allowed
to invoke this tool?"* — PALADIN evaluates tool selections against **task intent
and capability requirements**, not just caller identity.

TRAC predicates verify that the *selected tool bundle* satisfies the
*mission's capability profile*: correct domain alignment, sufficient action
coverage, and adequate confidence — independently of who the caller is.
This means a legitimately authorized agent can still be denied if its tool
choice is wrong *for the task at hand*, closing a class of over-privilege
vulnerabilities that identity-only models cannot detect.

> **Limitations:** LLM panel is three OpenAI generations plus a blind Claude Opus 4.8
> probe, a purpose-built dataset (ASTRA), and no production-scale latency benchmarks.
> See the paper for full discussion.

---

## Governance Pipeline Architecture

Every tool-use request flows through a **layered pipeline** where each layer independently
evaluates and can deny or redirect the request:

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Phase I:        │     │  Phase II:        │     │  Phase III:       │
│  Pre-LLM         │ ──► │  LLM Inference    │ ──► │  Post-LLM Logic   │
│                  │     │                   │     │                   │
│  ◆ SPIFFE ID     │     │  ◆ Tool Selection │     │  ◆ RBAC Check     │
│  ◆ Registry      │     │  ◆ Confidence     │     │  ◆ ABAC Rules     │
│  ◆ Allowlist     │     │  ◆ Justification  │     │  ◆ Fact Extraction│
│                  │     │                   │     │  ◆ TRAC Rules  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                    ┌────┴────┐
                                                    ▼         ▼
                                                  ALLOW     DENY
```

### Unified Decision Engine (6-Step Sequence)
1. **Pre-LLM Gate (SPIFFE Identity & Transport Allowlist)** — Verifies the caller identity format, existence, and mTLS access restrictions
2. **LLM Inference** — Autonomous selection/validation of MCP tools. Skipped if Step 1 fails
3. **Fact Extraction (Tool Audit & Intent Decomposition)** — Tool-centric action classification, capability mapping, and intent inference
4. **RBAC** — Evaluates the caller's allowed/denied permissions against requested MCP tools
5. **ABAC Baseline** — Evaluates attribute-based contextual rules (role, MCP, action, confidence, risk)
6. **TRAC Reasoning** — Executes formal predicate logic to reach the final authoritative decision

---

## Policy Layers (9 Configurable)

| # | Layer | File | Purpose |
|---|-------|------|---------|
| 1 | SPIFFE Registry | `policies/spiffe_registry.json` | Cryptographic identity anchoring (SPIFFE IDs → roles, trust, clearance) |
| 2 | Transport Allowlist | `policies/spiffe_allowlist.json` | Pre-LLM identity gate (DENY if caller not in allowlist) |
| 3 | MCP Attributes | `policies/mcp_attributes.yaml` | Risk metadata per MCP server (risk levels, compliance tiers) |
| 4 | RBAC Policies | `policies/rbac.yaml` | Role → permitted MCP domains/tools mapping |
| 5 | ABAC Rules | `policies/abac_rules.yaml` | Context-aware attribute-based access enforcement |
| 6 | Domain Catalog | `policies/domain_capability_catalog.json` | Tool → domain → action classification |
| 7 | Capability Ontology | `policies/domain_capability_ontology.json` | Intent → required capabilities mapping |
| 8 | Heuristic Logic | `policies/heuristic_policy.json` | Fallback tool classification via verb-prefix matching |
| 9 | TRAC Rules | `policies/trac_rules.yaml` | Task-relational safety predicates (4 rules: 2 enforcing + 2 advisory) |

All policies are editable through the **Policy Studio** UI — changes take effect immediately.

---

## Key Terminology

| Term | Meaning in PALADIN |
|------|-------------------|
| **Persona** | A SPIFFE-authenticated agent identity with a role, trust score, and clearance level (e.g., DevOps Agent, Finance Agent, Research Agent). The 6 personas are the callers evaluated in experiments. |
| **Domain** | An MCP server and its tool catalog (e.g., Atlassian, GitHub, Hummingbot, MongoDB). Each domain represents a distinct set of capabilities available for tool selection. |

The evaluation matrix consists of **1,157 tasks × 6 personas = 6,942 decisions**.

---

## SPIRE Integration

PALADIN includes a fully integrated **SPIFFE/SPIRE** identity infrastructure:

- **Cloud (Azure Container Apps):** SPIRE server + agent run as a sidecar in the container.
  Real X.509 SVIDs are issued automatically on startup — 6 workload identities registered.
- **Local (Docker Compose):** Deploy SPIRE via `python deploy_spire.py` or the UI's
  built-in **🔧 SPIRE Deployment Controls** button.
- **Simulation mode:** If SPIRE is unavailable, the system operates with simulated identities.

Trust domain: `spiffe://demo.local`

---

## Application Sections

| Section | Description |
|---------|-------------|
| **🏠 Home** | Problem statement, research questions, novelty overview, pipeline architecture, and system state |
| **🛡️ Policy Studio** | Configure and inspect all 9 policy layers. SPIFFE Identity Registry with live SPIRE controls |
| **🤖 MCP Domain Explorer** | Browse the MCP server catalog — tools, descriptions, and capability scope per domain (e.g., Atlassian, GitHub, MongoDB) |
| **🔍 ASTRA Task Explorer** | Explore the evaluation dataset (1,157 tasks × 6 personas). Filter by MCP domain, task category, and match tag |
| **🔮 Prediction Lab** | Run individual tasks through the full governance pipeline with detailed predicate traces |
| **🧪 Experiment Lab** | Run batch experiments (E1–E4) with ablation analysis. OPA baseline comparison tab. AI-powered assessment |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Language** | Python 3.12 |
| **LLM** | OpenAI, Anthropic, Google Gemini, or Azure AI Foundry (configurable model/deployment) |
| **Identity** | SPIFFE/SPIRE 1.9.6 |
| **Policy Baseline** | OPA-equivalent Rego engine (Python) |
| **Hosting** | Azure Container Apps |
| **Container** | Docker (Python 3.12-slim + SPIRE sidecar) |
| **CI/CD** | Azure Developer CLI (`azd`) |

---

## Installation & Usage

### Prerequisites
- Python 3.10+ (3.12 recommended)
- An API key for your chosen LLM provider (for LLM-powered experiments)
- Docker (optional — for local SPIRE infrastructure)

### Quick Start

```bash
# 1. Clone the repository (anonymised mirror for review)
git clone https://anonymous.4open.science/r/PALADIN-anon
cd PALADIN-anon

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Start SPIRE identity infrastructure
python deploy_spire.py

# 4. Run the application
streamlit run main.py
```

### LLM provider settings

In the sidebar's **LLM Settings**, choose **OpenAI**, **Anthropic (Claude)**,
**Google (Gemini)**, or **Azure AI Foundry**, then enter your own **API key** in
the password field. Select a model preset or **Custom model...** to enter any
text/JSON-capable model ID available to your provider account.

For Azure, select a deployment or **Custom deployment...**, and supply your
Azure OpenAI-compatible **resource endpoint** and **API version**. Use the
deployment name from your resource, not the underlying model ID.

User-entered keys and provider-specific choices are held only in the current app session
and survive switching providers; they are not written to configuration files
or the process environment. Delete the password-field value to clear your key.
Prediction Lab, its AI post-mortem, and Experiment LLM Lab use these same settings.
Changing settings makes no API calls. **Test connection** sends a small request
to the selected model and may incur provider charges.

**Only user-provided keys are accepted.** There is no server-credentials option
and no API-key fallback to `.env` or process environment variables. Leaving the
key empty disables LLM actions for that provider; browsing the app remains
available. Programmatic `LLMProvider` callers must also pass an explicit key.

### Azure Deployment

The application is configured for Azure Container Apps with SPIRE sidecar:

```bash
# Login to Azure
az login

# Deploy (provisions infrastructure + builds + deploys)
azd up
```

Infrastructure is defined in `infra/` using Bicep (ACR + Container Apps Environment + Container App + Log Analytics).

---

## File Structure

```
├── main.py                          # Streamlit entrypoint
├── Dockerfile                       # Container image (Python + SPIRE sidecar)
├── azure.yaml                       # Azure Developer CLI config
├── requirements.txt                 # Python dependencies
├── deploy_spire.py                  # Local SPIRE deployment script
├── .gitattributes                   # Line ending rules (LF for .sh, .conf)
│
├── app/
│   ├── models/                      # Data models (AstraTask, MCPPersona/Domain)
│   ├── services/                    # Core logic
│   │   ├── decision_engine.py       # 6-step unified decision pipeline
│   │   ├── prediction_service.py    # LLM inference service
│   │   ├── policy_loader.py         # Policy file loader
│   │   ├── rbac_service.py          # RBAC evaluation
│   │   ├── abac_rule_service.py     # ABAC evaluation
│   │   ├── tsphol_rule_service.py   # TRAC rule engine (loads trac_rules.yaml)
│   │   ├── spiffe_registry_service.py   # SPIFFE identity management
│   │   ├── spiffe_workload_service.py   # SPIRE integration (sidecar + Docker)
│   │   ├── capability_inference_service.py  # Capability mapping
│   │   ├── opa_comparison_engine.py     # OPA baseline comparison
│   │   └── ...
│   └── ui/                          # Streamlit UI components
│       ├── home.py                  # Home page
│       ├── policy_studio.py         # Policy Studio + SPIRE controls
│       ├── prediction_lab.py        # Prediction Lab
│       ├── experiment_lab.py        # Experiment Lab + OPA tab
│       └── ...
│
├── policies/                        # Configurable policy files (JSON/YAML)
│   ├── spiffe_registry.json
│   ├── spiffe_allowlist.json
│   ├── rbac.yaml
│   ├── abac_rules.yaml
│   ├── trac_rules.yaml
│   ├── mcp_attributes.yaml
│   ├── domain_capability_catalog.json
│   ├── domain_capability_ontology.json
│   ├── heuristic_policy.json
│   └── rego/                        # Generic OPA/Rego evaluators (rules-as-data)
│       ├── rbac.rego               #   evaluates data.policies (from rbac.yaml)
│       ├── abac.rego               #   evaluates data.rules    (from abac_rules.yaml)
│       ├── tsphol.rego             #   evaluates data.tsphol_rules (from trac_rules.yaml)
│       └── data/                   #   verb lexicon as OPA data (action_lexicon.json)
│
├── datasets/                        # ASTRA dataset & generated artifacts
│   ├── astra_tasks.json             # 1,157 tasks across 8 MCP domains
│   └── mcp_personas.json            # NOTE: "personas" here = SPIFFE agent identities (not MCP domains)
│
├── infra/                           # Azure infrastructure (Bicep)
│   ├── main.bicep                   # Subscription-scoped entry point
│   ├── main.parameters.json
│   ├── modules/
│   │   └── web.bicep                # Container Apps + ACR + Log Analytics
│   └── spire/
│       ├── sidecar/                 # Cloud SPIRE configs
│       │   ├── server.conf
│       │   ├── agent.conf
│       │   └── start-spire.sh       # Container startup orchestrator
│       ├── server/server.conf       # Local Docker SPIRE server config
│       ├── agent/agent.conf         # Local Docker SPIRE agent config
│       ├── docker-compose.yml       # Local SPIRE containers
│       └── register_workloads.sh
│
└── results/                         # Runtime output logs
    ├── prediction_logs.jsonl
    ├── decision_logs.jsonl
    └── policy_changes.jsonl
```

---

## License

This project is a research platform for studying layered policy governance over LLM-based agentic systems.

© 2026 PALADIN — Policy-Aware Layered Agentic Decision Intelligence
