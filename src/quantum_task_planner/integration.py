"""Integration adapters for external systems and frameworks."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import logging

from .core import Task, TaskState, QuantumPriority, QuantumTaskPlanner
from .exceptions import QuantumTaskPlannerError


@dataclass
class IntegrationConfig:
    """Configuration for external integrations."""
    
    # API Configuration
    enable_rest_api: bool = True
    api_host: str = "localhost"
    api_port: int = 8000
    api_version: str = "v1"
    
    # Webhook Configuration
    enable_webhooks: bool = False
    webhook_endpoints: List[str] = None
    webhook_timeout: float = 30.0
    
    # Message Queue Configuration
    enable_message_queue: bool = False
    queue_type: str = "redis"  # redis, rabbitmq, kafka
    queue_host: str = "localhost"
    queue_port: int = 6379
    
    # Database Configuration
    enable_persistence: bool = False
    database_url: str = "sqlite:///quantum_tasks.db"
    
    # Monitoring Integration
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    
    def __post_init__(self):
        if self.webhook_endpoints is None:
            self.webhook_endpoints = []


class TaskSerializer:
    """Serializer for quantum tasks."""
    
    @staticmethod
    def task_to_dict(task: Task) -> Dict[str, Any]:
        """Convert task to dictionary representation.
        
        Args:
            task: Task to serialize
            
        Returns:
            Dictionary representation
        """
        return {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "priority": task.priority.value,
            "state": task.state.value,
            "amplitude": task.amplitude,
            "phase": task.phase,
            "coherence_time": task.coherence_time,
            "entangled_tasks": list(task.entangled_tasks),
            "created_at": task.created_at.isoformat(),
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "estimated_duration": task.estimated_duration,
            "dependencies": list(task.dependencies),
            "metadata": task.metadata,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "progress": task.progress,
            "probability": task.probability,
            "is_coherent": task.is_coherent,
            "is_executable": task.is_executable
        }
    
    @staticmethod
    def dict_to_task(data: Dict[str, Any]) -> Task:
        """Convert dictionary to task object.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Task object
        """
        # Parse dates
        created_at = datetime.fromisoformat(data["created_at"])
        due_date = datetime.fromisoformat(data["due_date"]) if data.get("due_date") else None
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        
        task = Task(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            priority=QuantumPriority(data["priority"]),
            state=TaskState(data["state"]),
            amplitude=data["amplitude"],
            phase=data["phase"],
            coherence_time=data["coherence_time"],
            entangled_tasks=set(data["entangled_tasks"]),
            created_at=created_at,
            due_date=due_date,
            estimated_duration=data["estimated_duration"],
            dependencies=set(data["dependencies"]),
            metadata=data["metadata"],
            started_at=started_at,
            completed_at=completed_at,
            progress=data["progress"]
        )
        
        return task


class EventBus:
    """Event bus for integration notifications."""
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger("quantum_planner.events")
    
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Subscribe to events.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Event handler function
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        self.logger.debug(f"Subscribed handler to {event_type} events")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from events.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                self.logger.debug(f"Unsubscribed handler from {event_type} events")
            except ValueError:
                pass
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publish event to subscribers.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type not in self.subscribers:
            return
        
        self.logger.debug(f"Publishing {event_type} event to {len(self.subscribers[event_type])} subscribers")
        
        # Notify all subscribers
        for handler in self.subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data)
                else:
                    handler(event_data)
            except Exception as e:
                self.logger.error(f"Event handler failed for {event_type}: {e}")


class WebhookNotifier:
    """Webhook notification system."""
    
    def __init__(self, config: IntegrationConfig, event_bus: EventBus):
        """Initialize webhook notifier.
        
        Args:
            config: Integration configuration
            event_bus: Event bus for notifications
        """
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger("quantum_planner.webhooks")
        
        # Subscribe to relevant events
        if self.config.enable_webhooks:
            self.event_bus.subscribe("task_created", self._handle_task_event)
            self.event_bus.subscribe("task_completed", self._handle_task_event)
            self.event_bus.subscribe("task_failed", self._handle_task_event)
            self.event_bus.subscribe("quantum_collapse", self._handle_quantum_event)
    
    async def _handle_task_event(self, event_data: Dict[str, Any]):
        """Handle task-related events.
        
        Args:
            event_data: Event data
        """
        if not self.config.webhook_endpoints:
            return
        
        # Send webhook notifications
        for endpoint in self.config.webhook_endpoints:
            await self._send_webhook(endpoint, event_data)
    
    async def _handle_quantum_event(self, event_data: Dict[str, Any]):
        """Handle quantum-related events.
        
        Args:
            event_data: Event data
        """
        if not self.config.webhook_endpoints:
            return
        
        # Add quantum-specific metadata
        quantum_data = {
            **event_data,
            "event_category": "quantum",
            "timestamp": datetime.now().isoformat()
        }
        
        for endpoint in self.config.webhook_endpoints:
            await self._send_webhook(endpoint, quantum_data)
    
    async def _send_webhook(self, endpoint: str, data: Dict[str, Any]):
        """Send webhook notification.
        
        Args:
            endpoint: Webhook URL
            data: Data to send
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)
                ) as response:
                    if response.status >= 400:
                        self.logger.warning(f"Webhook {endpoint} returned {response.status}")
                    else:
                        self.logger.debug(f"Webhook sent to {endpoint}")
        
        except Exception as e:
            self.logger.error(f"Failed to send webhook to {endpoint}: {e}")


class RESTAPIAdapter:
    """REST API adapter for HTTP integration."""
    
    def __init__(self, planner: QuantumTaskPlanner, config: IntegrationConfig):
        """Initialize REST API adapter.
        
        Args:
            planner: Quantum task planner instance
            config: Integration configuration
        """
        self.planner = planner
        self.config = config
        self.logger = logging.getLogger("quantum_planner.api")
        self.app = None
    
    def create_app(self):
        """Create FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
        except ImportError:
            raise QuantumTaskPlannerError(
                "FastAPI not installed. Install with: pip install fastapi uvicorn",
                error_code="DEPENDENCY_MISSING"
            )
        
        app = FastAPI(
            title="Quantum Task Planner API",
            description="REST API for quantum-inspired task planning",
            version=self.config.api_version
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Pydantic models
        class TaskCreate(BaseModel):
            name: str
            description: str = ""
            priority: str = "medium"
            estimated_duration: float = 1.0
            dependencies: List[str] = []
            metadata: Dict[str, Any] = {}
        
        class TaskResponse(BaseModel):
            id: str
            name: str
            state: str
            probability: float
            created_at: str
        
        # API endpoints
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @app.get("/tasks", response_model=List[TaskResponse])
        async def get_tasks():
            """Get all tasks."""
            tasks = []
            for task in self.planner.tasks.values():
                tasks.append(TaskResponse(
                    id=task.id,
                    name=task.name,
                    state=task.state.value,
                    probability=task.probability,
                    created_at=task.created_at.isoformat()
                ))
            return tasks
        
        @app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get specific task."""
            task = self.planner.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return TaskSerializer.task_to_dict(task)
        
        @app.post("/tasks", response_model=TaskResponse)
        async def create_task(task_data: TaskCreate):
            """Create new task."""
            try:
                priority = QuantumPriority(task_data.priority.lower())
                
                task = self.planner.create_task(
                    name=task_data.name,
                    description=task_data.description,
                    priority=priority,
                    estimated_duration=task_data.estimated_duration,
                    dependencies=task_data.dependencies,
                    metadata=task_data.metadata
                )
                
                return TaskResponse(
                    id=task.id,
                    name=task.name,
                    state=task.state.value,
                    probability=task.probability,
                    created_at=task.created_at.isoformat()
                )
            
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.post("/tasks/{task_id}/execute")
        async def execute_task(task_id: str):
            """Execute specific task."""
            try:
                success = await self.planner.execute_task(task_id)
                return {"success": success, "task_id": task_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/system/state")
        async def get_system_state():
            """Get system state."""
            return self.planner.get_system_state()
        
        @app.post("/quantum/cycle")
        async def run_quantum_cycle():
            """Run quantum planning cycle."""
            try:
                result = await self.planner.run_quantum_cycle()
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        self.app = app
        return app
    
    async def start_server(self):
        """Start API server."""
        if not self.config.enable_rest_api:
            return
        
        try:
            import uvicorn
        except ImportError:
            raise QuantumTaskPlannerError(
                "uvicorn not installed. Install with: pip install uvicorn",
                error_code="DEPENDENCY_MISSING"
            )
        
        if not self.app:
            self.create_app()
        
        config = uvicorn.Config(
            app=self.app,
            host=self.config.api_host,
            port=self.config.api_port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        self.logger.info(f"Starting API server on {self.config.api_host}:{self.config.api_port}")
        await server.serve()


class MessageQueueAdapter:
    """Message queue integration adapter."""
    
    def __init__(self, planner: QuantumTaskPlanner, config: IntegrationConfig):
        """Initialize message queue adapter.
        
        Args:
            planner: Quantum task planner instance
            config: Integration configuration
        """
        self.planner = planner
        self.config = config
        self.logger = logging.getLogger("quantum_planner.queue")
        self.connection = None
    
    async def connect(self):
        """Connect to message queue."""
        if not self.config.enable_message_queue:
            return
        
        if self.config.queue_type == "redis":
            await self._connect_redis()
        elif self.config.queue_type == "rabbitmq":
            await self._connect_rabbitmq()
        else:
            raise QuantumTaskPlannerError(
                f"Unsupported queue type: {self.config.queue_type}",
                error_code="UNSUPPORTED_QUEUE_TYPE"
            )
    
    async def _connect_redis(self):
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            
            self.connection = redis.Redis(
                host=self.config.queue_host,
                port=self.config.queue_port,
                decode_responses=True
            )
            
            # Test connection
            await self.connection.ping()
            self.logger.info(f"Connected to Redis at {self.config.queue_host}:{self.config.queue_port}")
        
        except ImportError:
            raise QuantumTaskPlannerError(
                "redis not installed. Install with: pip install redis",
                error_code="DEPENDENCY_MISSING"
            )
        except Exception as e:
            raise QuantumTaskPlannerError(
                f"Failed to connect to Redis: {e}",
                error_code="QUEUE_CONNECTION_FAILED"
            )
    
    async def _connect_rabbitmq(self):
        """Connect to RabbitMQ."""
        try:
            import aio_pika
            
            connection_url = f"amqp://guest:guest@{self.config.queue_host}:{self.config.queue_port}/"
            self.connection = await aio_pika.connect_robust(connection_url)
            
            self.logger.info(f"Connected to RabbitMQ at {self.config.queue_host}:{self.config.queue_port}")
        
        except ImportError:
            raise QuantumTaskPlannerError(
                "aio-pika not installed. Install with: pip install aio-pika",
                error_code="DEPENDENCY_MISSING"
            )
        except Exception as e:
            raise QuantumTaskPlannerError(
                f"Failed to connect to RabbitMQ: {e}",
                error_code="QUEUE_CONNECTION_FAILED"
            )
    
    async def publish_task_event(self, event_type: str, task: Task):
        """Publish task event to queue.
        
        Args:
            event_type: Type of event
            task: Task object
        """
        if not self.connection:
            return
        
        message_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "task": TaskSerializer.task_to_dict(task)
        }
        
        if self.config.queue_type == "redis":
            await self._publish_redis(f"quantum_tasks.{event_type}", message_data)
        elif self.config.queue_type == "rabbitmq":
            await self._publish_rabbitmq(f"quantum_tasks.{event_type}", message_data)
    
    async def _publish_redis(self, channel: str, data: Dict[str, Any]):
        """Publish message to Redis."""
        try:
            await self.connection.publish(channel, json.dumps(data))
            self.logger.debug(f"Published message to Redis channel: {channel}")
        except Exception as e:
            self.logger.error(f"Failed to publish to Redis: {e}")
    
    async def _publish_rabbitmq(self, routing_key: str, data: Dict[str, Any]):
        """Publish message to RabbitMQ."""
        try:
            import aio_pika
            
            channel = await self.connection.channel()
            
            message = aio_pika.Message(
                json.dumps(data).encode(),
                content_type="application/json"
            )
            
            await channel.default_exchange.publish(
                message,
                routing_key=routing_key
            )
            
            self.logger.debug(f"Published message to RabbitMQ: {routing_key}")
        except Exception as e:
            self.logger.error(f"Failed to publish to RabbitMQ: {e}")
    
    async def disconnect(self):
        """Disconnect from message queue."""
        if self.connection:
            if self.config.queue_type == "redis":
                await self.connection.close()
            elif self.config.queue_type == "rabbitmq":
                await self.connection.close()
            
            self.logger.info("Disconnected from message queue")


class IntegrationManager:
    """Main integration management system."""
    
    def __init__(self, planner: QuantumTaskPlanner, config: Optional[IntegrationConfig] = None):
        """Initialize integration manager.
        
        Args:
            planner: Quantum task planner instance
            config: Integration configuration
        """
        self.planner = planner
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger("quantum_planner.integration")
        
        # Integration components
        self.event_bus = EventBus()
        self.webhook_notifier = WebhookNotifier(self.config, self.event_bus)
        self.api_adapter = RESTAPIAdapter(planner, self.config)
        self.queue_adapter = MessageQueueAdapter(planner, self.config)
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers for planner events."""
        # Task lifecycle events
        self.event_bus.subscribe("task_created", self._handle_task_created)
        self.event_bus.subscribe("task_completed", self._handle_task_completed)
        self.event_bus.subscribe("task_failed", self._handle_task_failed)
        
        # Quantum events
        self.event_bus.subscribe("quantum_collapse", self._handle_quantum_collapse)
        self.event_bus.subscribe("entanglement_created", self._handle_entanglement_created)
    
    async def _handle_task_created(self, event_data: Dict[str, Any]):
        """Handle task creation event."""
        task = event_data.get("task")
        if task and self.config.enable_message_queue:
            await self.queue_adapter.publish_task_event("created", task)
    
    async def _handle_task_completed(self, event_data: Dict[str, Any]):
        """Handle task completion event."""
        task = event_data.get("task")
        if task and self.config.enable_message_queue:
            await self.queue_adapter.publish_task_event("completed", task)
    
    async def _handle_task_failed(self, event_data: Dict[str, Any]):
        """Handle task failure event."""
        task = event_data.get("task")
        if task and self.config.enable_message_queue:
            await self.queue_adapter.publish_task_event("failed", task)
    
    async def _handle_quantum_collapse(self, event_data: Dict[str, Any]):
        """Handle quantum collapse event."""
        self.logger.info(f"Quantum collapse detected: {event_data}")
    
    async def _handle_entanglement_created(self, event_data: Dict[str, Any]):
        """Handle entanglement creation event."""
        self.logger.info(f"Entanglement created: {event_data}")
    
    async def start(self):
        """Start all integrations."""
        self.logger.info("Starting integration manager")
        
        # Start message queue connection
        if self.config.enable_message_queue:
            await self.queue_adapter.connect()
        
        # Start API server (runs in background)
        if self.config.enable_rest_api:
            import asyncio
            asyncio.create_task(self.api_adapter.start_server())
        
        self.logger.info("Integration manager started successfully")
    
    async def stop(self):
        """Stop all integrations."""
        self.logger.info("Stopping integration manager")
        
        # Disconnect from message queue
        if self.config.enable_message_queue:
            await self.queue_adapter.disconnect()
        
        self.logger.info("Integration manager stopped")
    
    async def notify_task_event(self, event_type: str, task: Task):
        """Notify integrations of task event.
        
        Args:
            event_type: Type of event
            task: Task object
        """
        event_data = {
            "event_type": event_type,
            "task": task,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.event_bus.publish(event_type, event_data)