"""Storage backends for audit trail data with encryption support."""

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    from google.cloud.exceptions import GoogleCloudError
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from .crypto import CryptographicEngine
from .exceptions import AuditTrailError, StorageError, IntegrityError

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    def __init__(self, crypto_engine: Optional[CryptographicEngine] = None,
                 encryption_password: Optional[str] = None):
        self.crypto = crypto_engine
        self.encryption_password = encryption_password
        self.encrypted_storage = crypto_engine is not None and encryption_password is not None
        
    @abstractmethod
    def store(self, key: str, data: Union[str, bytes, Dict[str, Any]], 
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data with optional encryption."""
        pass
        
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Union[str, bytes]]:
        """Retrieve and optionally decrypt data."""
        pass
        
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if data exists for key."""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data for key."""
        pass
        
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix filter."""
        pass
    
    async def store_encrypted(self, key: str, data: Union[str, bytes, Dict[str, Any]], 
                             crypto_engine: Any) -> bool:
        """Store data with encryption using provided crypto engine."""
        # Convert data to JSON string if dict
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        # Store using regular store method (LocalStorage handles encryption internally)
        return self.store(key, data)
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files with given prefix (async version of list_keys)."""
        return self.list_keys(prefix)
    
    def _prepare_data_for_storage(self, data: Union[str, bytes, Dict[str, Any]]) -> bytes:
        """Prepare data for storage with optional encryption."""
        # Convert to bytes
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
            
        # Encrypt if configured
        if self.encrypted_storage:
            encrypted_bundle, _ = self.crypto.encrypt_data(data_bytes, self.encryption_password)
            return encrypted_bundle
        else:
            return data_bytes
            
    def _process_retrieved_data(self, data_bytes: bytes) -> Union[str, bytes]:
        """Process retrieved data with optional decryption."""
        if self.encrypted_storage:
            try:
                decrypted_data = self.crypto.decrypt_data(data_bytes, self.encryption_password)
                return decrypted_data.decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to decrypt data: {e}")
                raise StorageError(
                    f"Data decryption failed: {e}",
                    operation="decrypt",
                    storage_backend=self.__class__.__name__
                )
        else:
            try:
                return data_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return data_bytes
                
    def verify_integrity(self, key: str, expected_hash: Optional[str] = None) -> bool:
        """Verify data integrity for stored data."""
        try:
            data = self.retrieve(key)
            if data is None:
                return False
                
            if expected_hash:
                if isinstance(data, str):
                    actual_hash = self.crypto.hash_data(data) if self.crypto else None
                    return actual_hash == expected_hash
                    
            return True
        except Exception as e:
            logger.error(f"Integrity verification failed for {key}: {e}")
            raise IntegrityError(
                f"Integrity verification failed: {e}",
                data_path=key,
                verification_method="hash_comparison"
            )


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str = "./audit_data", 
                 crypto_engine: Optional[CryptographicEngine] = None,
                 encryption_password: Optional[str] = None):
        super().__init__(crypto_engine, encryption_password)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized local storage at: {self.base_path}")
        
    def store(self, key: str, data: Union[str, bytes, Dict[str, Any]], 
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data to local filesystem."""
        try:
            file_path = self.base_path / key
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            data_bytes = self._prepare_data_for_storage(data)
            
            with open(file_path, 'wb') as f:
                f.write(data_bytes)
                
            # Store metadata if provided
            if metadata:
                metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            logger.debug(f"Stored data to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data locally: {e}")
            return False
            
    def retrieve(self, key: str) -> Optional[Union[str, bytes]]:
        """Retrieve data from local filesystem."""
        try:
            file_path = self.base_path / key
            if not file_path.exists():
                return None
                
            with open(file_path, 'rb') as f:
                data_bytes = f.read()
                
            return self._process_retrieved_data(data_bytes)
            
        except Exception as e:
            logger.error(f"Failed to retrieve data locally: {e}")
            return None
            
    def exists(self, key: str) -> bool:
        """Check if file exists locally."""
        file_path = self.base_path / key
        return file_path.exists()
        
    def delete(self, key: str) -> bool:
        """Delete local file."""
        try:
            file_path = self.base_path / key
            if file_path.exists():
                file_path.unlink()
                
                # Delete metadata file if exists
                metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
                if metadata_path.exists():
                    metadata_path.unlink()
                    
                logger.debug(f"Deleted file: {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file locally: {e}")
            return False
            
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix filter."""
        try:
            keys = []
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith('.meta'):
                    relative_path = file_path.relative_to(self.base_path)
                    key = str(relative_path)
                    if key.startswith(prefix):
                        keys.append(key)
            return sorted(keys)
            
        except Exception as e:
            logger.error(f"Failed to list keys locally: {e}")
            return []


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(self, bucket_name: str, prefix: str = "audit-trail/",
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = "us-east-1",
                 crypto_engine: Optional[CryptographicEngine] = None,
                 encryption_password: Optional[str] = None):
        if not BOTO3_AVAILABLE:
            raise AuditTrailError("boto3 not available. Install with: pip install boto3")
            
        super().__init__(crypto_engine, encryption_password)
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Initialized S3 storage: s3://{bucket_name}/{self.prefix}")
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to initialize S3 storage: {e}")
            raise AuditTrailError(f"S3 initialization failed: {e}")
            
    def _get_s3_key(self, key: str) -> str:
        """Get full S3 key with prefix."""
        return f"{self.prefix}{key}"
        
    def store(self, key: str, data: Union[str, bytes, Dict[str, Any]], 
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data to S3."""
        try:
            s3_key = self._get_s3_key(key)
            data_bytes = self._prepare_data_for_storage(data)
            
            # Prepare metadata
            s3_metadata = {'timestamp': datetime.utcnow().isoformat()}
            if metadata:
                # S3 metadata must be strings
                for k, v in metadata.items():
                    s3_metadata[k] = str(v)
                    
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data_bytes,
                Metadata=s3_metadata,
                ServerSideEncryption='AES256'  # S3 server-side encryption
            )
            
            logger.debug(f"Stored data to S3: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to store data to S3: {e}")
            return False
            
    def retrieve(self, key: str) -> Optional[Union[str, bytes]]:
        """Retrieve data from S3."""
        try:
            s3_key = self._get_s3_key(key)
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            data_bytes = response['Body'].read()
            return self._process_retrieved_data(data_bytes)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.error(f"Failed to retrieve data from S3: {e}")
            return None
            
    def exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        try:
            s3_key = self._get_s3_key(key)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Failed to check S3 object existence: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete object from S3."""
        try:
            s3_key = self._get_s3_key(key)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.debug(f"Deleted S3 object: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to delete S3 object: {e}")
            return False
            
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys in S3 with optional prefix filter."""
        try:
            full_prefix = self._get_s3_key(prefix)
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            keys = []
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Remove the prefix to get relative key
                        key = obj['Key'][len(self.prefix):]
                        keys.append(key)
                        
            return sorted(keys)
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to list S3 keys: {e}")
            return []


class GCPStorage(StorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(self, bucket_name: str, prefix: str = "audit-trail/",
                 credentials_path: Optional[str] = None,
                 crypto_engine: Optional[CryptographicEngine] = None,
                 encryption_password: Optional[str] = None):
        if not GCP_AVAILABLE:
            raise AuditTrailError("google-cloud-storage not available. Install with: pip install google-cloud-storage")
            
        super().__init__(crypto_engine, encryption_password)
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        
        try:
            if credentials_path:
                self.client = gcs.Client.from_service_account_json(credentials_path)
            else:
                self.client = gcs.Client()
                
            self.bucket = self.client.bucket(bucket_name)
            
            # Test connection
            self.bucket.exists()
            logger.info(f"Initialized GCS storage: gs://{bucket_name}/{self.prefix}")
            
        except GoogleCloudError as e:
            logger.error(f"Failed to initialize GCS storage: {e}")
            raise AuditTrailError(f"GCS initialization failed: {e}")
            
    def _get_blob_name(self, key: str) -> str:
        """Get full blob name with prefix."""
        return f"{self.prefix}{key}"
        
    def store(self, key: str, data: Union[str, bytes, Dict[str, Any]], 
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data to GCS."""
        try:
            blob_name = self._get_blob_name(key)
            blob = self.bucket.blob(blob_name)
            data_bytes = self._prepare_data_for_storage(data)
            
            # Set metadata
            if metadata:
                blob.metadata = {k: str(v) for k, v in metadata.items()}
                
            blob.upload_from_string(data_bytes)
            logger.debug(f"Stored data to GCS: gs://{self.bucket_name}/{blob_name}")
            return True
            
        except GoogleCloudError as e:
            logger.error(f"Failed to store data to GCS: {e}")
            return False
            
    def retrieve(self, key: str) -> Optional[Union[str, bytes]]:
        """Retrieve data from GCS."""
        try:
            blob_name = self._get_blob_name(key)
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                return None
                
            data_bytes = blob.download_as_bytes()
            return self._process_retrieved_data(data_bytes)
            
        except GoogleCloudError as e:
            logger.error(f"Failed to retrieve data from GCS: {e}")
            return None
            
    def exists(self, key: str) -> bool:
        """Check if blob exists in GCS."""
        try:
            blob_name = self._get_blob_name(key)
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except GoogleCloudError as e:
            logger.error(f"Failed to check GCS blob existence: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete blob from GCS."""
        try:
            blob_name = self._get_blob_name(key)
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.debug(f"Deleted GCS blob: gs://{self.bucket_name}/{blob_name}")
            return True
            
        except GoogleCloudError as e:
            logger.error(f"Failed to delete GCS blob: {e}")
            return False
            
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys in GCS with optional prefix filter."""
        try:
            full_prefix = self._get_blob_name(prefix)
            blobs = self.client.list_blobs(self.bucket, prefix=full_prefix)
            
            keys = []
            for blob in blobs:
                # Remove the prefix to get relative key
                key = blob.name[len(self.prefix):]
                keys.append(key)
                
            return sorted(keys)
            
        except GoogleCloudError as e:
            logger.error(f"Failed to list GCS keys: {e}")
            return []


class StorageManager:
    """Manages multiple storage backends with fallback support."""
    
    def __init__(self, primary_backend: StorageBackend, 
                 fallback_backends: Optional[List[StorageBackend]] = None):
        self.primary_backend = primary_backend
        self.fallback_backends = fallback_backends or []
        
    def store(self, key: str, data: Union[str, bytes, Dict[str, Any]], 
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store data with fallback support."""
        # Try primary backend first
        if self.primary_backend.store(key, data, metadata):
            return True
            
        # Try fallback backends
        for backend in self.fallback_backends:
            if backend.store(key, data, metadata):
                logger.warning(f"Used fallback backend for storing: {key}")
                return True
                
        logger.error(f"Failed to store data with all backends: {key}")
        return False
        
    def retrieve(self, key: str) -> Optional[Union[str, bytes]]:
        """Retrieve data with fallback support."""
        # Try primary backend first
        result = self.primary_backend.retrieve(key)
        if result is not None:
            return result
            
        # Try fallback backends
        for backend in self.fallback_backends:
            result = backend.retrieve(key)
            if result is not None:
                logger.warning(f"Used fallback backend for retrieving: {key}")
                return result
                
        return None
        
    def exists(self, key: str) -> bool:
        """Check if data exists in any backend."""
        if self.primary_backend.exists(key):
            return True
            
        for backend in self.fallback_backends:
            if backend.exists(key):
                return True
                
        return False


def create_storage_backend(backend_type: str, **config) -> StorageBackend:
    """Factory function to create storage backends."""
    backends = {
        'local': LocalStorage,
        's3': S3Storage,
        'gcs': GCPStorage,
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unsupported backend type: {backend_type}")
        
    return backends[backend_type](**config)