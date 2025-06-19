# PHML Agentic RAG System

An intelligent customer service agent for Police Health Maintenance Limited (PHML) Nigeria, featuring RAG (Retrieval-Augmented Generation) capabilities with ReAct agent architecture and human agent routing.

## 🚀 Key Features

### 1. **RAG System**
- **Vector-based knowledge retrieval** using LlamaIndex
- **Gemini embeddings** for semantic search
- **Document indexing** from the `data/` directory
- **Contextual responses** based on PHML documentation

### 2. **ReAct Agent Architecture**
- **Tool-based reasoning** with function calling
- **Multi-step problem solving** capabilities
- **Structured decision making** for complex queries
- **Verbose logging** for transparency

### 3. **Human Agent Routing**
- **Intelligent escalation** for complex/sensitive queries
- **Ticket generation** with unique IDs (PHML-XXXXX format)
- **Priority classification** (low, medium, high, urgent)
- **Department routing** (medical, billing, claims, enrollment, general)

## 🏥 About PHML

Police Health Maintenance Limited is Nigeria's number one HMO provider for:
- Nigeria Police Force personnel
- Their family members
- Quality, comprehensive healthcare under GIFSHIP plan
- Affordable and accessible healthcare services

## 🛠️ Technical Architecture

### Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   ReAct Agent    │────│  Human Routing  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐             │
         └──────────────│   RAG System     │─────────────┘
                        │  (LlamaIndex)    │
                        └──────────────────┘
```

### Tools Available to Agent
1. **`search_phml_knowledge`** - Searches the RAG knowledge base
2. **`route_to_human_agent`** - Routes complex queries to human agents

### Routing Triggers
The agent automatically routes to human agents when encountering:
- Complex medical emergencies or urgent health concerns
- Billing disputes or payment issues
- Claims processing problems or denials
- Enrollment or registration difficulties
- Provider network issues or hospital access problems
- Customer complaints about service quality
- Requests for manager or supervisor
- Legal or compliance matters
- Sensitive medical information requiring privacy
- Any query it cannot adequately address

## 📁 File Structure

```
phml-ai/
├── phml_agent.py           # Main Streamlit application
├── sample_agent.py         # Reference implementation
├── test_phml_agent.py      # Test script for agent functionality
├── data/
│   └── sample.md          # PHML knowledge base
└── README_PHML_Agent.md   # This documentation
```

## 🚀 Usage

### Running the Streamlit App
```bash
streamlit run phml_agent.py
```

### Testing the Agent
```bash
python test_phml_agent.py
```

### Example Interactions

**Simple Knowledge Query:**
```
User: "What is PHML?"
Agent: [Uses search_phml_knowledge tool to provide information about PHML services]
```

**Complex Query Requiring Human Agent:**
```
User: "I want to file a complaint about my claim being denied"
Agent: [Uses route_to_human_agent tool]
Response: "I've created ticket PHML-12345 and routed your request to our claims team..."
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY` - Required for Gemini LLM and embeddings

### Customization Options
- **System Prompt**: Modify the `system_prompt` variable to adjust agent behavior
- **Routing Logic**: Update routing conditions in the system prompt
- **UI Layout**: Customize Streamlit interface in the UI section
- **Tools**: Add new tools by creating functions and adding them to the agent

## 🎯 Benefits

1. **24/7 Availability** - Instant responses to customer queries
2. **Intelligent Escalation** - Automatic routing of complex issues
3. **Consistent Service** - Standardized responses based on PHML knowledge
4. **Scalability** - Handles multiple customers simultaneously
5. **Cost Effective** - Reduces human agent workload for routine queries
6. **Audit Trail** - Complete conversation and routing history

## 🔄 Workflow

1. **User Input** → Streamlit chat interface
2. **Agent Processing** → ReAct agent analyzes query
3. **Tool Selection** → Chooses between knowledge search or human routing
4. **Response Generation** → Provides appropriate response
5. **Session Management** → Tracks routing status and conversation history

## 📊 Monitoring & Analytics

The system tracks:
- Routing frequency and reasons
- Query types and patterns
- Response quality and user satisfaction
- Human agent workload distribution

## 🔮 Future Enhancements

- Integration with actual ticketing systems
- Real-time human agent availability
- Multi-language support
- Voice interface capabilities
- Advanced analytics dashboard
- Integration with PHML's existing systems

---

**Developed for PHML Nigeria** - Enhancing healthcare accessibility through intelligent automation.
