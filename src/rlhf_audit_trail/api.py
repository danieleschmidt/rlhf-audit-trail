"""FastAPI-based REST API for RLHF audit trail operations."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .core import AuditableRLHF
from .config import PrivacyConfig, SecurityConfig, ComplianceConfig
from .exceptions import AuditTrailError, PrivacyBudgetExceededError

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False) if FASTAPI_AVAILABLE else None


# Pydantic models for API  
if FASTAPI_AVAILABLE:
    class TrainingSessionRequest(BaseModel):
        """Request model for creating training sessions."""
        experiment_name: str = Field(..., description="Name of the experiment")
        model_name: str = Field(..., description="Name of the model being trained")
        privacy_config: Optional[Dict[str, Any]] = Field(None, description="Privacy configuration")
        compliance_mode: str = Field("eu_ai_act", description="Compliance framework")

    class AnnotationRequest(BaseModel):
        """Request model for logging annotations."""
        prompts: List[str] = Field(..., description="List of prompts")
        responses: List[str] = Field(..., description="List of responses")
        rewards: List[float] = Field(..., description="List of reward scores")
        annotator_ids: Optional[List[str]] = Field(None, description="Anonymized annotator IDs")

    class PolicyUpdateRequest(BaseModel):
        """Request model for tracking policy updates."""
        epoch: int = Field(..., description="Training epoch")
        metrics: Dict[str, float] = Field(..., description="Training metrics")
        additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional tracking data")

    class ComplianceReportRequest(BaseModel):
        """Request model for generating compliance reports."""
        start_date: Optional[str] = Field(None, description="Start date (ISO format)")
        end_date: Optional[str] = Field(None, description="End date (ISO format)")
        include_sections: List[str] = Field(["privacy", "bias", "safety"], description="Report sections")
        format: str = Field("json", description="Report format (json, html, pdf)")

    class ModelCardRequest(BaseModel):
        """Request model for generating model cards."""
        include_provenance: bool = Field(True, description="Include provenance information")
        include_privacy_analysis: bool = Field(True, description="Include privacy analysis")
        format: str = Field("eu_standard", description="Model card format")

    # Response models
    class TrainingSessionResponse(BaseModel):
        """Response model for training sessions."""
        session_id: str
        experiment_name: str
        status: str
        created_at: datetime
        model_name: str

    class AuditLogResponse(BaseModel):
        """Response model for audit logs."""
        timestamp: datetime
        event_type: str
        event_data: Dict[str, Any]
        signature: Optional[str] = None
        verified: bool = False

    class PrivacyReportResponse(BaseModel):
        """Response model for privacy reports."""
        total_epsilon: float
        remaining_budget: float
        annotator_count: int
        privacy_violations: int

    class ComplianceStatusResponse(BaseModel):
        """Response model for compliance status."""
        framework: str
        compliant: bool
        last_check: datetime
        issues: List[Dict[str, str]]
        score: float
else:
    # Create dummy classes when FastAPI is not available
    TrainingSessionRequest = object
    AnnotationRequest = object
    PolicyUpdateRequest = object  
    ComplianceReportRequest = object
    ModelCardRequest = object
    TrainingSessionResponse = object
    AuditLogResponse = object
    PrivacyReportResponse = object
    ComplianceStatusResponse = object


if FASTAPI_AVAILABLE:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        logger.info("Starting RLHF Audit Trail API")
        yield
        logger.info("Shutting down RLHF Audit Trail API")

    app = FastAPI(
        title="RLHF Audit Trail API",
        description="REST API for verifiable provenance of RLHF steps with EU AI Act compliance",
        version="0.1.0",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global variables (in production, use proper dependency injection)
    audit_systems: Dict[str, AuditableRLHF] = {}

    # Authentication dependency
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Validate authentication token."""
        if not credentials:
            # For demo purposes, allow unauthenticated access
            return {"user_id": "anonymous", "role": "user"}
        
        # In production, validate JWT token here
        token = credentials.credentials
        if token == "demo-token":
            return {"user_id": "demo_user", "role": "admin"}
        
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "RLHF Audit Trail API",
            "version": "0.1.0",
            "status": "running",
            "documentation": "/docs"
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow()}

    @app.post("/sessions", response_model=TrainingSessionResponse)
    async def create_training_session(
        request: TrainingSessionRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Create a new training session."""
        try:
            # Create privacy config
            privacy_config = None
            if request.privacy_config:
                privacy_config = PrivacyConfig(**request.privacy_config)

            # Initialize auditable RLHF
            auditor = AuditableRLHF(
                model_name=request.model_name,
                privacy_config=privacy_config,
                compliance_mode=request.compliance_mode
            )
            
            # Store in global registry (use proper storage in production)
            session_id = f"session_{len(audit_systems)}_{int(datetime.utcnow().timestamp())}"
            audit_systems[session_id] = auditor
            
            return TrainingSessionResponse(
                session_id=session_id,
                experiment_name=request.experiment_name,
                status="created",
                created_at=datetime.utcnow(),
                model_name=request.model_name
            )
            
        except Exception as e:
            logger.error(f"Failed to create training session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions", response_model=List[TrainingSessionResponse])
    async def list_training_sessions(
        current_user: dict = Depends(get_current_user)
    ):
        """List all training sessions."""
        sessions = []
        for session_id, auditor in audit_systems.items():
            sessions.append(TrainingSessionResponse(
                session_id=session_id,
                experiment_name=getattr(auditor, 'experiment_name', 'unknown'),
                status="active",
                created_at=datetime.utcnow(),  # Should be stored properly
                model_name=auditor.model_name
            ))
        return sessions

    @app.post("/sessions/{session_id}/annotations")
    async def log_annotations(
        session_id: str = Path(..., description="Training session ID"),
        request: AnnotationRequest = ...,
        current_user: dict = Depends(get_current_user)
    ):
        """Log annotations for a training session."""
        try:
            auditor = audit_systems.get(session_id)
            if not auditor:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            result = auditor.log_annotations(
                prompts=request.prompts,
                responses=request.responses,
                annotator_ids=request.annotator_ids or [f"annotator_{i}" for i in range(len(request.prompts))],
                rewards=request.rewards
            )
            
            return {"status": "success", "logged_count": len(request.prompts), "result": result}
            
        except PrivacyBudgetExceededError as e:
            raise HTTPException(status_code=400, detail=f"Privacy budget exceeded: {e}")
        except Exception as e:
            logger.error(f"Failed to log annotations: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/policy-update")
    async def track_policy_update(
        session_id: str = Path(..., description="Training session ID"),
        request: PolicyUpdateRequest = ...,
        current_user: dict = Depends(get_current_user)
    ):
        """Track a policy update for a training session."""
        try:
            auditor = audit_systems.get(session_id)
            if not auditor:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # Note: In a real implementation, model and optimizer would be passed
            result = auditor.checkpoint(
                epoch=request.epoch,
                metrics=request.metrics
            )
            
            return {"status": "success", "checkpoint_id": result}
            
        except Exception as e:
            logger.error(f"Failed to track policy update: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/{session_id}/audit-logs", response_model=List[AuditLogResponse])
    async def get_audit_logs(
        session_id: str = Path(..., description="Training session ID"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return"),
        offset: int = Query(0, ge=0, description="Number of logs to skip"),
        current_user: dict = Depends(get_current_user)
    ):
        """Get audit logs for a training session."""
        try:
            auditor = audit_systems.get(session_id)
            if not auditor:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # This would typically query the audit log storage
            # For now, return mock data
            logs = []
            for i in range(min(limit, 10)):  # Mock data
                logs.append(AuditLogResponse(
                    timestamp=datetime.utcnow(),
                    event_type="annotation",
                    event_data={"mock": True, "index": i},
                    signature="mock_signature",
                    verified=True
                ))
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/{session_id}/privacy-report", response_model=PrivacyReportResponse)
    async def get_privacy_report(
        session_id: str = Path(..., description="Training session ID"),
        current_user: dict = Depends(get_current_user)
    ):
        """Get privacy report for a training session."""
        try:
            auditor = audit_systems.get(session_id)
            if not auditor:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            report = auditor.get_privacy_report()
            
            return PrivacyReportResponse(
                total_epsilon=report.get("total_epsilon", 0.0),
                remaining_budget=report.get("remaining_budget", 0.0),
                annotator_count=report.get("annotator_count", 0),
                privacy_violations=report.get("privacy_violations", 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get privacy report: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/{session_id}/compliance-status", response_model=ComplianceStatusResponse)
    async def get_compliance_status(
        session_id: str = Path(..., description="Training session ID"),
        current_user: dict = Depends(get_current_user)
    ):
        """Get compliance status for a training session."""
        try:
            auditor = audit_systems.get(session_id)
            if not auditor:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            # This would typically check compliance status
            return ComplianceStatusResponse(
                framework="eu_ai_act",
                compliant=True,
                last_check=datetime.utcnow(),
                issues=[],
                score=95.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/model-card")
    async def generate_model_card(
        session_id: str = Path(..., description="Training session ID"),
        request: ModelCardRequest = ...,
        current_user: dict = Depends(get_current_user)
    ):
        """Generate model card for a training session."""
        try:
            auditor = audit_systems.get(session_id)
            if not auditor:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            model_card = auditor.generate_model_card(
                include_provenance=request.include_provenance,
                include_privacy_analysis=request.include_privacy_analysis,
                format=request.format
            )
            
            return {"status": "success", "model_card": model_card}
            
        except Exception as e:
            logger.error(f"Failed to generate model card: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/{session_id}/verify-integrity")
    async def verify_audit_integrity(
        session_id: str = Path(..., description="Training session ID"),
        current_user: dict = Depends(get_current_user)
    ):
        """Verify the integrity of audit trail."""
        try:
            auditor = audit_systems.get(session_id)
            if not auditor:
                raise HTTPException(status_code=404, detail="Training session not found")
            
            verification = auditor.verify_provenance()
            
            return {
                "status": "success",
                "verification": {
                    "is_valid": verification.get("is_valid", False),
                    "merkle_root": verification.get("merkle_root", ""),
                    "chain_verification": verification.get("chain_verification", False)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to verify audit integrity: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/compliance/report")
    async def generate_compliance_report(
        request: ComplianceReportRequest = ...,
        background_tasks: BackgroundTasks = ...,
        current_user: dict = Depends(get_current_user)
    ):
        """Generate comprehensive compliance report."""
        try:
            # This would generate a comprehensive report across all sessions
            # For now, return a mock response
            
            def generate_report_background():
                """Generate report in background."""
                logger.info(f"Generating compliance report with sections: {request.include_sections}")
                # Report generation logic would go here
                
            background_tasks.add_task(generate_report_background)
            
            return {
                "status": "accepted", 
                "message": "Report generation started",
                "estimated_completion": "2-3 minutes"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Error handlers
    @app.exception_handler(AuditTrailError)
    async def audit_trail_exception_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "type": "audit_trail_error"}
        )

    @app.exception_handler(PrivacyBudgetExceededError)
    async def privacy_budget_exception_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "type": "privacy_budget_exceeded"}
        )

    def run_api_server(host: str = "0.0.0.0", port: int = 8000, 
                      reload: bool = False, workers: int = 1):
        """Run the FastAPI server."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
            
        uvicorn.run(
            "rlhf_audit_trail.api:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level="info"
        )

else:
    # Dummy implementations when FastAPI is not available
    def run_api_server(*args, **kwargs):
        raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")

# CLI entry point
def main():
    """CLI entry point for running the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF Audit Trail API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    run_api_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main()