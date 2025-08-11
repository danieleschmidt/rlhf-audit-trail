"""
Multi-region deployment and data residency management for RLHF audit trail.
Global-first implementation with compliance and data sovereignty.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcp_storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from .exceptions import StorageError, ComplianceViolationError, ConfigurationError
from .internationalization import Region

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    LOCAL = "local"


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    region: Region
    cloud_provider: CloudProvider
    primary_zone: str
    backup_zones: List[str]
    data_classification: str
    encryption_key_region: str
    compliance_frameworks: List[str]
    data_residency_strict: bool = False
    cross_border_transfer_allowed: bool = True
    backup_retention_days: int = 90
    log_retention_years: int = 7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class DataResidencyRule:
    """Data residency and sovereignty rules."""
    region: Region
    allowed_storage_regions: List[str]
    prohibited_regions: List[str]
    requires_encryption: bool
    requires_local_keys: bool
    cross_border_approval_required: bool
    data_categories: List[str]  # Types of data this rule applies to
    compliance_framework: str
    enforcement_level: str  # "strict", "advisory", "flexible"


class RegionManager:
    """Manages multi-region deployment configurations."""
    
    def __init__(self):
        """Initialize region manager."""
        self.regions: Dict[Region, RegionConfig] = {}
        self.data_residency_rules: List[DataResidencyRule] = []
        self.active_regions: List[Region] = []
        
        self._setup_default_configurations()
        self._setup_data_residency_rules()
    
    def _setup_default_configurations(self):
        """Setup default region configurations."""
        default_configs = {
            Region.US: RegionConfig(
                region=Region.US,
                cloud_provider=CloudProvider.AWS,
                primary_zone="us-east-1",
                backup_zones=["us-west-2", "us-central-1"],
                data_classification="public",
                encryption_key_region="us-east-1",
                compliance_frameworks=["NIST_AI_RMF", "SOC2", "FedRAMP"],
                cross_border_transfer_allowed=True,
                log_retention_years=7
            ),
            
            Region.EU: RegionConfig(
                region=Region.EU,
                cloud_provider=CloudProvider.AWS,
                primary_zone="eu-west-1",
                backup_zones=["eu-central-1", "eu-west-3"],
                data_classification="personal",
                encryption_key_region="eu-west-1",
                compliance_frameworks=["EU_AI_ACT", "GDPR", "ISO_27001"],
                data_residency_strict=True,
                cross_border_transfer_allowed=False,
                log_retention_years=3
            ),
            
            Region.CN: RegionConfig(
                region=Region.CN,
                cloud_provider=CloudProvider.ALIBABA,
                primary_zone="cn-beijing",
                backup_zones=["cn-shanghai", "cn-hangzhou"],
                data_classification="restricted",
                encryption_key_region="cn-beijing",
                compliance_frameworks=["PIPL", "CYBERSECURITY_LAW"],
                data_residency_strict=True,
                cross_border_transfer_allowed=False,
                log_retention_years=3
            ),
            
            Region.JP: RegionConfig(
                region=Region.JP,
                cloud_provider=CloudProvider.AWS,
                primary_zone="ap-northeast-1",
                backup_zones=["ap-northeast-3"],
                data_classification="personal",
                encryption_key_region="ap-northeast-1",
                compliance_frameworks=["PERSONAL_INFO_PROTECTION_ACT"],
                cross_border_transfer_allowed=True,
                log_retention_years=5
            ),
            
            Region.AU: RegionConfig(
                region=Region.AU,
                cloud_provider=CloudProvider.AWS,
                primary_zone="ap-southeast-2",
                backup_zones=["ap-southeast-4"],
                data_classification="personal",
                encryption_key_region="ap-southeast-2",
                compliance_frameworks=["PRIVACY_ACT", "AI_ETHICS_PRINCIPLES"],
                cross_border_transfer_allowed=True,
                log_retention_years=7
            ),
            
            Region.CA: RegionConfig(
                region=Region.CA,
                cloud_provider=CloudProvider.AWS,
                primary_zone="ca-central-1",
                backup_zones=["ca-west-1"],
                data_classification="personal",
                encryption_key_region="ca-central-1",
                compliance_frameworks=["PIPEDA", "AIDA"],
                cross_border_transfer_allowed=True,
                log_retention_years=7
            ),
            
            Region.GB: RegionConfig(
                region=Region.GB,
                cloud_provider=CloudProvider.AWS,
                primary_zone="eu-west-2",
                backup_zones=["eu-west-1"],
                data_classification="personal",
                encryption_key_region="eu-west-2",
                compliance_frameworks=["UK_GDPR", "AI_WHITE_PAPER"],
                data_residency_strict=True,
                cross_border_transfer_allowed=False,
                log_retention_years=6
            )
        }
        
        self.regions.update(default_configs)
    
    def _setup_data_residency_rules(self):
        """Setup data residency and sovereignty rules."""
        rules = [
            # EU GDPR Rules
            DataResidencyRule(
                region=Region.EU,
                allowed_storage_regions=["eu-west-1", "eu-central-1", "eu-west-3", "eu-north-1"],
                prohibited_regions=["us-east-1", "us-west-2", "cn-beijing", "ap-south-1"],
                requires_encryption=True,
                requires_local_keys=True,
                cross_border_approval_required=True,
                data_categories=["personal_data", "biometric_data", "training_data"],
                compliance_framework="GDPR",
                enforcement_level="strict"
            ),
            
            # China Data Security Rules
            DataResidencyRule(
                region=Region.CN,
                allowed_storage_regions=["cn-beijing", "cn-shanghai", "cn-hangzhou", "cn-shenzhen"],
                prohibited_regions=["us-east-1", "eu-west-1", "ap-northeast-1"],
                requires_encryption=True,
                requires_local_keys=True,
                cross_border_approval_required=True,
                data_categories=["personal_data", "critical_data", "ai_models"],
                compliance_framework="PIPL",
                enforcement_level="strict"
            ),
            
            # US Federal Rules
            DataResidencyRule(
                region=Region.US,
                allowed_storage_regions=["us-east-1", "us-west-2", "us-central-1", "ca-central-1"],
                prohibited_regions=["cn-beijing", "ru-central-1"],
                requires_encryption=True,
                requires_local_keys=False,
                cross_border_approval_required=False,
                data_categories=["government_data", "defense_data"],
                compliance_framework="FedRAMP",
                enforcement_level="advisory"
            ),
            
            # UK Post-Brexit Rules
            DataResidencyRule(
                region=Region.GB,
                allowed_storage_regions=["eu-west-2", "eu-west-1", "us-east-1"],
                prohibited_regions=["cn-beijing", "ru-central-1"],
                requires_encryption=True,
                requires_local_keys=True,
                cross_border_approval_required=True,
                data_categories=["personal_data", "financial_data"],
                compliance_framework="UK_GDPR",
                enforcement_level="strict"
            )
        ]
        
        self.data_residency_rules = rules
    
    def add_region(self, region_config: RegionConfig):
        """Add a new region configuration."""
        self.regions[region_config.region] = region_config
        logger.info(f"Added region configuration: {region_config.region.value}")
    
    def get_region_config(self, region: Region) -> Optional[RegionConfig]:
        """Get configuration for a specific region."""
        return self.regions.get(region)
    
    def validate_data_residency(self, region: Region, data_category: str, 
                               storage_region: str) -> Tuple[bool, List[str]]:
        """
        Validate data residency requirements.
        
        Returns:
            Tuple of (is_valid, violations)
        """
        violations = []
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.data_residency_rules
            if rule.region == region and data_category in rule.data_categories
        ]
        
        if not applicable_rules:
            return True, []  # No specific rules, allow
        
        for rule in applicable_rules:
            # Check prohibited regions
            if storage_region in rule.prohibited_regions:
                violations.append(
                    f"Storage in {storage_region} prohibited by {rule.compliance_framework}"
                )
            
            # Check allowed regions (if strict)
            if (rule.enforcement_level == "strict" and 
                storage_region not in rule.allowed_storage_regions):
                violations.append(
                    f"Storage in {storage_region} not explicitly allowed by {rule.compliance_framework}"
                )
        
        return len(violations) == 0, violations
    
    def get_recommended_storage_regions(self, region: Region, 
                                      data_category: str) -> List[str]:
        """Get recommended storage regions for data category."""
        applicable_rules = [
            rule for rule in self.data_residency_rules
            if rule.region == region and data_category in rule.data_categories
        ]
        
        if not applicable_rules:
            # No specific rules, return region's default zones
            config = self.get_region_config(region)
            if config:
                return [config.primary_zone] + config.backup_zones
            return []
        
        # Intersection of all allowed regions from applicable rules
        allowed_regions = None
        for rule in applicable_rules:
            if allowed_regions is None:
                allowed_regions = set(rule.allowed_storage_regions)
            else:
                allowed_regions = allowed_regions.intersection(rule.allowed_storage_regions)
        
        return list(allowed_regions) if allowed_regions else []


class MultiRegionStorageManager:
    """Manages storage across multiple regions with data residency compliance."""
    
    def __init__(self, region_manager: RegionManager):
        """Initialize multi-region storage manager."""
        self.region_manager = region_manager
        self.storage_clients: Dict[str, Any] = {}
        self.encryption_keys: Dict[str, str] = {}
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize cloud storage clients for each region."""
        for region, config in self.region_manager.regions.items():
            try:
                if config.cloud_provider == CloudProvider.AWS and AWS_AVAILABLE:
                    self._init_aws_client(region, config)
                elif config.cloud_provider == CloudProvider.GCP and GCP_AVAILABLE:
                    self._init_gcp_client(region, config)
                elif config.cloud_provider == CloudProvider.AZURE and AZURE_AVAILABLE:
                    self._init_azure_client(region, config)
                else:
                    logger.warning(f"Storage client not available for {region.value}")
            except Exception as e:
                logger.error(f"Failed to initialize storage client for {region.value}: {e}")
    
    def _init_aws_client(self, region: Region, config: RegionConfig):
        """Initialize AWS S3 client."""
        try:
            session = boto3.Session()
            s3_client = session.client('s3', region_name=config.primary_zone)
            self.storage_clients[region.value] = {
                'type': 'aws_s3',
                'client': s3_client,
                'bucket_prefix': f"rlhf-audit-{region.value.lower()}"
            }
            logger.info(f"AWS S3 client initialized for {region.value}")
        except Exception as e:
            logger.error(f"AWS S3 client initialization failed for {region.value}: {e}")
    
    def _init_gcp_client(self, region: Region, config: RegionConfig):
        """Initialize GCP Cloud Storage client."""
        try:
            client = gcp_storage.Client()
            self.storage_clients[region.value] = {
                'type': 'gcp_storage',
                'client': client,
                'bucket_prefix': f"rlhf-audit-{region.value.lower()}"
            }
            logger.info(f"GCP Storage client initialized for {region.value}")
        except Exception as e:
            logger.error(f"GCP Storage client initialization failed for {region.value}: {e}")
    
    def _init_azure_client(self, region: Region, config: RegionConfig):
        """Initialize Azure Blob Storage client."""
        try:
            # Would use Azure connection string from environment
            connection_string = os.getenv(f"AZURE_STORAGE_CONNECTION_STRING_{region.value}")
            if connection_string:
                blob_service = BlobServiceClient.from_connection_string(connection_string)
                self.storage_clients[region.value] = {
                    'type': 'azure_blob',
                    'client': blob_service,
                    'container_prefix': f"rlhf-audit-{region.value.lower()}"
                }
                logger.info(f"Azure Blob Storage client initialized for {region.value}")
        except Exception as e:
            logger.error(f"Azure client initialization failed for {region.value}: {e}")
    
    async def store_data(self, region: Region, data_category: str, 
                        key: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Store data in appropriate region with compliance validation.
        
        Returns:
            Storage location URI
        """
        config = self.region_manager.get_region_config(region)
        if not config:
            raise ConfigurationError(f"No configuration found for region: {region.value}")
        
        # Validate data residency
        is_valid, violations = self.region_manager.validate_data_residency(
            region, data_category, config.primary_zone
        )
        
        if not is_valid:
            raise ComplianceViolationError(
                f"Data residency violation for {data_category} in {region.value}",
                framework="DATA_RESIDENCY",
                violations=violations
            )
        
        # Get storage client
        storage_info = self.storage_clients.get(region.value)
        if not storage_info:
            raise StorageError(
                f"No storage client available for region: {region.value}",
                storage_backend=config.cloud_provider.value
            )
        
        # Encrypt data if required
        encrypted_data = data
        if config.data_classification in ["personal", "restricted"] or data_category == "personal_data":
            encrypted_data = self._encrypt_data(data, region)
        
        # Store data based on provider
        try:
            if storage_info['type'] == 'aws_s3':
                return await self._store_aws_s3(storage_info, key, encrypted_data, metadata)
            elif storage_info['type'] == 'gcp_storage':
                return await self._store_gcp(storage_info, key, encrypted_data, metadata)
            elif storage_info['type'] == 'azure_blob':
                return await self._store_azure(storage_info, key, encrypted_data, metadata)
            else:
                raise StorageError(f"Unsupported storage type: {storage_info['type']}")
        
        except Exception as e:
            logger.error(f"Storage operation failed for {region.value}: {e}")
            raise StorageError(
                f"Failed to store data in {region.value}: {str(e)}",
                storage_backend=config.cloud_provider.value,
                operation="store"
            )
    
    async def _store_aws_s3(self, storage_info: Dict, key: str, 
                           data: Any, metadata: Optional[Dict]) -> str:
        """Store data in AWS S3."""
        client = storage_info['client']
        bucket_name = f"{storage_info['bucket_prefix']}-{datetime.now().year}"
        
        # Ensure bucket exists
        try:
            client.head_bucket(Bucket=bucket_name)
        except ClientError:
            client.create_bucket(Bucket=bucket_name)
        
        # Prepare data
        data_bytes = json.dumps(data).encode('utf-8') if not isinstance(data, bytes) else data
        
        # Upload with metadata
        extra_args = {}
        if metadata:
            extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
        
        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=data_bytes,
            ServerSideEncryption='AES256',
            **extra_args
        )
        
        return f"s3://{bucket_name}/{key}"
    
    async def _store_gcp(self, storage_info: Dict, key: str, 
                        data: Any, metadata: Optional[Dict]) -> str:
        """Store data in GCP Cloud Storage."""
        client = storage_info['client']
        bucket_name = f"{storage_info['bucket_prefix']}-{datetime.now().year}"
        
        # Get or create bucket
        try:
            bucket = client.bucket(bucket_name)
            if not bucket.exists():
                bucket.create()
        except Exception as e:
            logger.error(f"GCP bucket operation failed: {e}")
            raise
        
        # Upload blob
        blob = bucket.blob(key)
        data_str = json.dumps(data) if not isinstance(data, (str, bytes)) else data
        blob.upload_from_string(data_str)
        
        # Set metadata
        if metadata:
            blob.metadata = metadata
            blob.patch()
        
        return f"gs://{bucket_name}/{key}"
    
    async def _store_azure(self, storage_info: Dict, key: str, 
                          data: Any, metadata: Optional[Dict]) -> str:
        """Store data in Azure Blob Storage."""
        client = storage_info['client']
        container_name = f"{storage_info['container_prefix']}-{datetime.now().year}"
        
        # Create container if needed
        try:
            client.create_container(container_name)
        except Exception:
            pass  # Container may already exist
        
        # Upload blob
        data_str = json.dumps(data) if not isinstance(data, (str, bytes)) else data
        blob_client = client.get_blob_client(container=container_name, blob=key)
        blob_client.upload_blob(data_str, overwrite=True, metadata=metadata)
        
        return f"https://{client.account_name}.blob.core.windows.net/{container_name}/{key}"
    
    def _encrypt_data(self, data: Any, region: Region) -> bytes:
        """Encrypt data for storage."""
        # Simplified encryption - in production, use proper key management
        import base64
        
        data_str = json.dumps(data) if not isinstance(data, str) else data
        encoded_data = base64.b64encode(data_str.encode('utf-8'))
        
        # Add region-specific encryption layer here
        return encoded_data
    
    async def retrieve_data(self, region: Region, storage_uri: str) -> Any:
        """Retrieve data from storage with compliance validation."""
        config = self.region_manager.get_region_config(region)
        if not config:
            raise ConfigurationError(f"No configuration found for region: {region.value}")
        
        storage_info = self.storage_clients.get(region.value)
        if not storage_info:
            raise StorageError(
                f"No storage client available for region: {region.value}",
                storage_backend=config.cloud_provider.value
            )
        
        try:
            # Parse storage URI and retrieve data
            if storage_uri.startswith('s3://'):
                return await self._retrieve_aws_s3(storage_info, storage_uri)
            elif storage_uri.startswith('gs://'):
                return await self._retrieve_gcp(storage_info, storage_uri)
            elif 'blob.core.windows.net' in storage_uri:
                return await self._retrieve_azure(storage_info, storage_uri)
            else:
                raise StorageError(f"Unsupported storage URI: {storage_uri}")
        
        except Exception as e:
            logger.error(f"Data retrieval failed from {region.value}: {e}")
            raise StorageError(
                f"Failed to retrieve data from {region.value}: {str(e)}",
                storage_backend=config.cloud_provider.value,
                operation="retrieve"
            )
    
    async def _retrieve_aws_s3(self, storage_info: Dict, uri: str) -> Any:
        """Retrieve data from AWS S3."""
        client = storage_info['client']
        
        # Parse S3 URI
        uri_parts = uri.replace('s3://', '').split('/', 1)
        bucket_name = uri_parts[0]
        key = uri_parts[1]
        
        response = client.get_object(Bucket=bucket_name, Key=key)
        data = response['Body'].read()
        
        return json.loads(data.decode('utf-8'))
    
    async def _retrieve_gcp(self, storage_info: Dict, uri: str) -> Any:
        """Retrieve data from GCP Cloud Storage."""
        client = storage_info['client']
        
        # Parse GCS URI
        uri_parts = uri.replace('gs://', '').split('/', 1)
        bucket_name = uri_parts[0]
        key = uri_parts[1]
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        data = blob.download_as_text()
        
        return json.loads(data)
    
    async def _retrieve_azure(self, storage_info: Dict, uri: str) -> Any:
        """Retrieve data from Azure Blob Storage."""
        # Parse Azure URI and retrieve - simplified implementation
        # In production, would parse the full URI properly
        blob_client = storage_info['client'].get_blob_client_from_url(uri)
        data = blob_client.download_blob().readall()
        
        return json.loads(data.decode('utf-8'))


class GeographicRoutingManager:
    """Manages geographic routing and latency optimization."""
    
    def __init__(self, region_manager: RegionManager):
        """Initialize geographic routing manager."""
        self.region_manager = region_manager
        self.latency_map: Dict[Tuple[Region, Region], float] = {}
        self._build_latency_map()
    
    def _build_latency_map(self):
        """Build estimated latency map between regions."""
        # Simplified latency estimates in milliseconds
        latencies = {
            # US internal
            (Region.US, Region.US): 10,
            (Region.US, Region.CA): 25,
            
            # Cross-Atlantic
            (Region.US, Region.EU): 120,
            (Region.US, Region.GB): 100,
            
            # Trans-Pacific
            (Region.US, Region.JP): 150,
            (Region.US, Region.CN): 180,
            (Region.US, Region.AU): 200,
            
            # Europe internal
            (Region.EU, Region.EU): 15,
            (Region.EU, Region.GB): 30,
            (Region.EU, Region.DE): 20,
            
            # Europe to Asia
            (Region.EU, Region.CN): 200,
            (Region.EU, Region.JP): 220,
            (Region.EU, Region.IN): 120,
            
            # Asia internal
            (Region.CN, Region.CN): 20,
            (Region.CN, Region.JP): 50,
            (Region.CN, Region.KR): 30,
            (Region.JP, Region.KR): 20,
            (Region.JP, Region.AU): 100,
        }
        
        # Make latency map symmetric
        for (region1, region2), latency in latencies.items():
            self.latency_map[(region1, region2)] = latency
            self.latency_map[(region2, region1)] = latency
    
    def get_optimal_region(self, user_region: Region, 
                          available_regions: List[Region],
                          data_category: str) -> Region:
        """
        Get optimal region for user considering latency and compliance.
        
        Args:
            user_region: User's geographic region
            available_regions: Regions where service is available
            data_category: Type of data being accessed
            
        Returns:
            Optimal region for the user
        """
        if not available_regions:
            raise ConfigurationError("No available regions provided")
        
        # Filter regions based on data residency rules
        compliant_regions = []
        for region in available_regions:
            is_valid, _ = self.region_manager.validate_data_residency(
                user_region, data_category, region.value
            )
            if is_valid:
                compliant_regions.append(region)
        
        if not compliant_regions:
            logger.warning(f"No compliant regions found for {user_region.value}, using available regions")
            compliant_regions = available_regions
        
        # Find region with lowest latency
        best_region = compliant_regions[0]
        best_latency = self.latency_map.get((user_region, best_region), float('inf'))
        
        for region in compliant_regions[1:]:
            latency = self.latency_map.get((user_region, region), float('inf'))
            if latency < best_latency:
                best_latency = latency
                best_region = region
        
        logger.info(f"Selected region {best_region.value} for user in {user_region.value} (latency: {best_latency}ms)")
        return best_region
    
    def get_failover_regions(self, primary_region: Region, 
                           user_region: Region) -> List[Region]:
        """Get failover regions ordered by preference."""
        config = self.region_manager.get_region_config(primary_region)
        if not config:
            return []
        
        # Get backup zones from config
        backup_regions = []
        for zone in config.backup_zones:
            # Map zone to region (simplified)
            if 'us-' in zone:
                backup_regions.append(Region.US)
            elif 'eu-' in zone:
                backup_regions.append(Region.EU)
            elif 'ap-northeast' in zone:
                backup_regions.append(Region.JP)
            elif 'ap-southeast' in zone:
                backup_regions.append(Region.AU)
            elif 'ca-' in zone:
                backup_regions.append(Region.CA)
        
        # Remove duplicates and sort by latency from user
        unique_regions = list(set(backup_regions))
        unique_regions.sort(key=lambda r: self.latency_map.get((user_region, r), float('inf')))
        
        return unique_regions


class MultiRegionDeploymentManager:
    """Main manager for multi-region deployments."""
    
    def __init__(self):
        """Initialize multi-region deployment manager."""
        self.region_manager = RegionManager()
        self.storage_manager = MultiRegionStorageManager(self.region_manager)
        self.routing_manager = GeographicRoutingManager(self.region_manager)
        
        # Deployment status
        self.active_deployments: Dict[Region, Dict[str, Any]] = {}
        
        logger.info("Multi-region deployment manager initialized")
    
    def deploy_to_region(self, region: Region, services: List[str]) -> Dict[str, Any]:
        """Deploy services to a specific region."""
        config = self.region_manager.get_region_config(region)
        if not config:
            raise ConfigurationError(f"No configuration for region: {region.value}")
        
        deployment_info = {
            "region": region.value,
            "cloud_provider": config.cloud_provider.value,
            "primary_zone": config.primary_zone,
            "services": services,
            "deployed_at": datetime.utcnow().isoformat(),
            "status": "deploying",
            "compliance_frameworks": config.compliance_frameworks
        }
        
        try:
            # Simulate deployment process
            logger.info(f"Deploying {services} to {region.value}")
            
            # Validate compliance requirements
            self._validate_deployment_compliance(region, services)
            
            # Deploy services (mock implementation)
            for service in services:
                self._deploy_service(region, service, config)
            
            deployment_info["status"] = "active"
            self.active_deployments[region] = deployment_info
            
            logger.info(f"Successfully deployed to {region.value}")
            return deployment_info
            
        except Exception as e:
            deployment_info["status"] = "failed"
            deployment_info["error"] = str(e)
            logger.error(f"Deployment to {region.value} failed: {e}")
            raise
    
    def _validate_deployment_compliance(self, region: Region, services: List[str]):
        """Validate deployment compliance requirements."""
        config = self.region_manager.get_region_config(region)
        
        # Check data residency requirements
        for service in services:
            if service == "audit_storage":
                is_valid, violations = self.region_manager.validate_data_residency(
                    region, "audit_data", config.primary_zone
                )
                if not is_valid:
                    raise ComplianceViolationError(
                        f"Cannot deploy {service} to {region.value}: {violations}",
                        framework="DATA_RESIDENCY",
                        violations=violations
                    )
        
        # Validate compliance framework coverage
        required_frameworks = {"audit_api": ["ISO_27001"], "ml_training": ["AI_ACT"]}
        
        for service in services:
            if service in required_frameworks:
                for framework in required_frameworks[service]:
                    if framework not in config.compliance_frameworks:
                        raise ComplianceViolationError(
                            f"Service {service} requires {framework} compliance in {region.value}",
                            framework=framework
                        )
    
    def _deploy_service(self, region: Region, service: str, config: RegionConfig):
        """Deploy individual service to region."""
        # Mock deployment - in production would use Infrastructure as Code
        deployment_steps = {
            "audit_api": ["create_compute", "configure_networking", "deploy_application"],
            "audit_storage": ["create_storage", "configure_encryption", "set_policies"],
            "ml_training": ["create_gpu_cluster", "install_frameworks", "configure_monitoring"],
            "dashboard": ["create_web_tier", "configure_cdn", "deploy_frontend"]
        }
        
        steps = deployment_steps.get(service, ["deploy_generic_service"])
        
        for step in steps:
            logger.debug(f"Executing {step} for {service} in {region.value}")
            # Simulate deployment time
            import time
            time.sleep(0.1)
        
        logger.info(f"Service {service} deployed successfully in {region.value}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get status of all regional deployments."""
        return {
            "total_regions": len(self.active_deployments),
            "healthy_regions": len([d for d in self.active_deployments.values() 
                                  if d["status"] == "active"]),
            "deployments": dict(self.active_deployments),
            "global_services": {
                "load_balancer": "active",
                "dns": "active", 
                "monitoring": "active"
            }
        }
    
    def handle_regional_failover(self, failed_region: Region, 
                               user_region: Region) -> Region:
        """Handle failover from failed region to backup."""
        logger.warning(f"Handling failover from {failed_region.value}")
        
        # Get failover regions
        failover_regions = self.routing_manager.get_failover_regions(
            failed_region, user_region
        )
        
        # Find first available failover region
        for region in failover_regions:
            if region in self.active_deployments:
                deployment = self.active_deployments[region]
                if deployment["status"] == "active":
                    logger.info(f"Failing over to {region.value}")
                    return region
        
        # No failover region found
        raise StorageError(f"No available failover region for {failed_region.value}")
    
    async def store_audit_data(self, user_region: Region, data_category: str,
                             key: str, data: Any) -> str:
        """Store audit data in appropriate region."""
        # Determine optimal storage region
        available_regions = list(self.active_deployments.keys())
        optimal_region = self.routing_manager.get_optimal_region(
            user_region, available_regions, data_category
        )
        
        # Store data with multi-region manager
        return await self.storage_manager.store_data(
            optimal_region, data_category, key, data
        )
    
    def get_regional_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary across all regions."""
        compliance_summary = {
            "frameworks_by_region": {},
            "data_residency_rules": len(self.region_manager.data_residency_rules),
            "regions_with_strict_residency": [],
            "cross_border_restrictions": []
        }
        
        for region, config in self.region_manager.regions.items():
            compliance_summary["frameworks_by_region"][region.value] = config.compliance_frameworks
            
            if config.data_residency_strict:
                compliance_summary["regions_with_strict_residency"].append(region.value)
            
            if not config.cross_border_transfer_allowed:
                compliance_summary["cross_border_restrictions"].append(region.value)
        
        return compliance_summary


# Global multi-region deployment manager
_global_deployment_manager: Optional[MultiRegionDeploymentManager] = None

def get_deployment_manager() -> MultiRegionDeploymentManager:
    """Get global multi-region deployment manager."""
    global _global_deployment_manager
    if _global_deployment_manager is None:
        _global_deployment_manager = MultiRegionDeploymentManager()
    return _global_deployment_manager