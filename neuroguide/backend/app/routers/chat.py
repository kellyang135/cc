# neuroguide/backend/app/routers/chat.py
"""Chat router for the NeuroGuide API."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.agent import NeuroGuideAgent

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Store active agents by conversation_id
active_agents: dict[str, NeuroGuideAgent] = {}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class VisualizationCommand(BaseModel):
    type: str
    data: dict


class ToolCallInfo(BaseModel):
    name: str
    arguments: dict
    result: str


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    visualizations: list[VisualizationCommand]
    tool_calls: list[ToolCallInfo]


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the NeuroGuide agent."""
    try:
        # Get or create agent for this conversation
        if request.conversation_id and request.conversation_id in active_agents:
            agent = active_agents[request.conversation_id]
        else:
            agent = NeuroGuideAgent(conversation_id=request.conversation_id)
            active_agents[agent.conversation_id] = agent

        # Process message
        result = await agent.chat(request.message)

        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            visualizations=[
                VisualizationCommand(type=v["type"], data=v["data"])
                for v in result["visualizations"]
            ],
            tool_calls=[
                ToolCallInfo(name=tc["name"], arguments=tc["arguments"], result=tc["result"])
                for tc in result["tool_calls"]
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_datasets():
    """List available datasets."""
    from app.services.datasets import DatasetService
    service = DatasetService()
    return service.list_datasets()
