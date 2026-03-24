# Agentic AI Summit 2026 - Submission Form

**Email:** yunkss@gmail.com

**Full Name:** Kyongsik Yun

**Organization / Affiliation:** NASA Jet Propulsion Laboratory (JPL)

**Job Title / Role:** Technologist

**LinkedIn / Personal Website / GitHub:** `[YOUR LINKEDIN]` / `[YOUR GITHUB]`

**Your X/Twitter handle:** *(left blank)*

**Speaker Bio:**
Kyongsik Yun is a Technologist at NASA's Jet Propulsion Laboratory, where he develops AI-powered tools for climate science. He is the creator of RCMES-MCP, an open-source MCP server that enables AI agents to analyze NASA's 38TB NEX-GDDP-CMIP6 climate dataset through natural language conversation. His work focuses on bridging the gap between massive Earth science datasets and accessible, agent-driven analysis — making climate data actionable for researchers, policymakers, and the public without requiring specialized programming skills.

**Presentation Title:**
RCMES-MCP: Giving AI Agents Access to 38TB of NASA Climate Data via the Model Context Protocol

**Presentation Format:** Technical Talk or Presentation

**Target Audience:** Engineers / Builders, Researchers

**Technical Level:** Intermediate

**Presentation Abstract (200 words max):**
Climate projections are critical for planning and policy, yet the NEX-GDDP-CMIP6 dataset — 38TB of downscaled CMIP6 projections at 0.25° resolution — remains inaccessible to most stakeholders due to its size and the programming expertise required to analyze it. We present RCMES-MCP, an open-source server built on the Model Context Protocol (MCP) that exposes NASA's Regional Climate Model Evaluation System as composable tools for AI agents. Through MCP, agents can load climate data from cloud-optimized Zarr stores on AWS S3, perform spatial and temporal subsetting, compute ETCCDI extreme indices (heatwaves, drought), calculate trends and climatologies, and generate publication-quality visualizations — all through natural language. The system uses a session-based architecture where tool operations are chainable via dataset IDs, enabling complex multi-step climate analyses. We also provide a web UI and REST API for interactive use. We discuss design decisions around tool granularity, session state management, and balancing agent autonomy with scientific rigor. RCMES-MCP demonstrates that MCP can transform domain-specific scientific software into agent-accessible infrastructure, lowering the barrier to climate data analysis from months of Python expertise to a single conversation.

**Key Takeaways for Attendees:**
1. How MCP enables AI agents to interact with large-scale scientific datasets (38TB+) without downloading data locally
2. Design patterns for building chainable, session-based MCP tool architectures for complex multi-step workflows
3. Practical strategies for wrapping existing scientific Python libraries (xarray, dask) as agent-accessible tools while preserving scientific rigor
4. Lessons learned from deploying an agentic climate analysis system — including tool granularity trade-offs and managing agent autonomy in scientific contexts
5. How this approach can be generalized to other NASA and Earth science datasets beyond climate projections

**If you have a paper associated with your presentation, please link here:** *(left blank — add if applicable)*

**Topic Areas:**
- Tool use / MCP / A2A protocols
- Agentic applications
- Agents for science, coding, web, research
- Agent frameworks & infrastructure
- Open-source agent ecosystems

**Has this work/paper been previously published or accepted at a conference or journal?** No

**If yes, where?** N/A

**Are you able to present in person on Aug 1-2 @UC Berkeley?** Yes

**Anything else you'd like the Summit program committee to know?**
RCMES-MCP is an open-source project developed at NASA JPL. A live demo is available at http://34.31.165.25:8502 showcasing the web interface for interactive climate analysis. The underlying MCP server is designed to work with any MCP-compatible AI client (Claude Desktop, Claude Code, etc.), making it a practical reference implementation for scientific agentic applications.
