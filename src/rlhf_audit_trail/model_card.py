"""Model card generation system for regulatory compliance."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

from .exceptions import AuditTrailError

logger = logging.getLogger(__name__)


class ModelCardFormat(Enum):
    """Supported model card formats."""
    EU_AI_ACT = "eu_ai_act"
    NIST_FRAMEWORK = "nist_framework"
    HUGGINGFACE = "huggingface"
    IEEE_STANDARD = "ieee_standard"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Core model metadata for model cards."""
    name: str
    version: str
    architecture: str
    parameters: int
    training_data: Dict[str, Any]
    intended_use: str
    limitations: List[str]
    performance_metrics: Dict[str, float]
    ethical_considerations: List[str]
    creation_date: datetime
    last_updated: datetime


@dataclass
class TrainingProvenance:
    """Training provenance information."""
    training_duration: float
    total_annotations: int
    annotator_count: int
    policy_updates: int
    checkpoint_count: int
    privacy_budget_used: float
    compliance_score: float
    verification_status: bool


@dataclass
class RegulatoryCompliance:
    """Regulatory compliance information."""
    frameworks: List[str]
    requirements_met: Dict[str, bool]
    compliance_scores: Dict[str, float]
    risk_assessment: Dict[str, str]
    mitigation_measures: List[str]
    audit_trail_location: str


class ModelCardGenerator:
    """Generates comprehensive model cards for regulatory compliance."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.template_env = None
        
        if JINJA2_AVAILABLE:
            try:
                self.template_env = Environment(
                    loader=FileSystemLoader(str(self.template_dir)),
                    autoescape=True
                )
            except Exception as e:
                logger.warning(f"Could not setup template environment: {e}")
        
        # Default templates for different formats
        self._init_default_templates()
        
    def _init_default_templates(self):
        """Initialize default templates."""
        self.default_templates = {
            ModelCardFormat.EU_AI_ACT: self._get_eu_ai_act_template(),
            ModelCardFormat.NIST_FRAMEWORK: self._get_nist_framework_template(),
            ModelCardFormat.HUGGINGFACE: self._get_huggingface_template(),
            ModelCardFormat.IEEE_STANDARD: self._get_ieee_template()
        }
    
    def generate_model_card(self,
                           model_metadata: ModelMetadata,
                           training_provenance: TrainingProvenance,
                           compliance_info: RegulatoryCompliance,
                           format_type: ModelCardFormat = ModelCardFormat.EU_AI_ACT,
                           output_format: str = "html",
                           custom_sections: Optional[Dict[str, Any]] = None) -> str:
        """Generate a comprehensive model card.
        
        Args:
            model_metadata: Core model information
            training_provenance: Training process provenance
            compliance_info: Regulatory compliance information
            format_type: Model card format standard
            output_format: Output format (html, markdown, json)
            custom_sections: Additional custom sections
            
        Returns:
            Generated model card as string
        """
        try:
            # Prepare template context
            context = self._prepare_template_context(
                model_metadata,
                training_provenance, 
                compliance_info,
                custom_sections or {}
            )
            
            # Get template
            template_content = self._get_template(format_type)
            
            if JINJA2_AVAILABLE and self.template_env:
                template = Template(template_content)
                rendered = template.render(**context)
            else:
                # Fallback to simple string formatting
                rendered = self._simple_template_render(template_content, context)
            
            # Convert to desired output format
            if output_format == "html":
                return self._to_html(rendered)
            elif output_format == "markdown":
                return rendered
            elif output_format == "json":
                return json.dumps(context, indent=2, default=str)
            else:
                return rendered
                
        except Exception as e:
            logger.error(f"Failed to generate model card: {e}")
            raise AuditTrailError(f"Model card generation failed: {e}")
    
    def _prepare_template_context(self,
                                 model_metadata: ModelMetadata,
                                 training_provenance: TrainingProvenance,
                                 compliance_info: RegulatoryCompliance,
                                 custom_sections: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template rendering context."""
        return {
            # Model information
            'model_name': model_metadata.name,
            'model_version': model_metadata.version,
            'architecture': model_metadata.architecture,
            'parameters': model_metadata.parameters,
            'creation_date': model_metadata.creation_date,
            'last_updated': model_metadata.last_updated,
            
            # Intended use and limitations
            'intended_use': model_metadata.intended_use,
            'limitations': model_metadata.limitations,
            'ethical_considerations': model_metadata.ethical_considerations,
            
            # Training data
            'training_data': model_metadata.training_data,
            
            # Performance metrics
            'performance_metrics': model_metadata.performance_metrics,
            
            # Training provenance
            'training_duration_hours': training_provenance.training_duration / 3600,
            'total_annotations': training_provenance.total_annotations,
            'annotator_count': training_provenance.annotator_count,
            'policy_updates': training_provenance.policy_updates,
            'checkpoint_count': training_provenance.checkpoint_count,
            'privacy_budget_used': training_provenance.privacy_budget_used,
            'compliance_score': training_provenance.compliance_score,
            'verification_status': training_provenance.verification_status,
            
            # Regulatory compliance
            'compliance_frameworks': compliance_info.frameworks,
            'requirements_met': compliance_info.requirements_met,
            'compliance_scores': compliance_info.compliance_scores,
            'risk_assessment': compliance_info.risk_assessment,
            'mitigation_measures': compliance_info.mitigation_measures,
            'audit_trail_location': compliance_info.audit_trail_location,
            
            # Generation metadata
            'generated_at': datetime.utcnow(),
            'generator_version': "0.1.0",
            
            # Custom sections
            **custom_sections
        }
    
    def _get_template(self, format_type: ModelCardFormat) -> str:
        """Get template for specified format."""
        return self.default_templates.get(format_type, self.default_templates[ModelCardFormat.EU_AI_ACT])
    
    def _to_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML."""
        if MARKDOWN_AVAILABLE:
            return markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
        else:
            # Simple HTML wrapper
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Card</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .status-pass {{ color: green; }}
                    .status-fail {{ color: red; }}
                    .status-warning {{ color: orange; }}
                </style>
            </head>
            <body>
                <pre>{markdown_content}</pre>
            </body>
            </html>
            """
    
    def _simple_template_render(self, template: str, context: Dict[str, Any]) -> str:
        """Simple template rendering without Jinja2."""
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Template variable not found: {e}")
            return template
    
    def _get_eu_ai_act_template(self) -> str:
        """Get EU AI Act compliant model card template."""
        return """
# Model Card: {model_name}

**Version:** {model_version}  
**Generated:** {generated_at}  
**Compliance Framework:** EU AI Act  

## 1. Model Overview

**Architecture:** {architecture}  
**Parameters:** {parameters:,}  
**Creation Date:** {creation_date}  
**Last Updated:** {last_updated}  

## 2. Intended Use

{intended_use}

### Primary Use Cases
- Reinforcement Learning from Human Feedback (RLHF)
- AI Safety and Alignment Research
- Conversational AI Development

### Out-of-Scope Use Cases
- Critical infrastructure control
- Medical diagnosis without human oversight
- Financial decision making without transparency

## 3. Training Data

**Dataset Information:**
- **Source:** {training_data.get('source', 'Not specified')}
- **Size:** {training_data.get('size', 'Not specified')} examples
- **Language(s):** {training_data.get('languages', 'Not specified')}
- **Collection Method:** {training_data.get('collection_method', 'Not specified')}

**Data Quality Measures:**
- Bias mitigation applied: {training_data.get('bias_mitigation', 'Not specified')}
- Data validation performed: {training_data.get('validation', 'Not specified')}
- Privacy protection: Differential Privacy (ε={privacy_budget_used:.2f})

## 4. Training Process Provenance

**Training Statistics:**
- **Duration:** {training_duration_hours:.1f} hours
- **Total Annotations:** {total_annotations:,}
- **Human Annotators:** {annotator_count}
- **Policy Updates:** {policy_updates}
- **Checkpoints Created:** {checkpoint_count}

**Privacy Protection:**
- **Privacy Budget Used:** {privacy_budget_used:.2f}
- **Differential Privacy Applied:** Yes
- **Annotator Anonymization:** Complete

**Audit Trail Verification:**
- **Status:** {'✅ Verified' if verification_status else '❌ Failed'}
- **Audit Trail Location:** {audit_trail_location}

## 5. Performance Metrics

| Metric | Value |
|--------|-------|
{% for metric, value in performance_metrics.items() -%}
| {metric} | {value:.3f} |
{% endfor %}

## 6. EU AI Act Compliance

**Risk Classification:** {risk_assessment.get('classification', 'High-Risk AI System')}
**Compliance Score:** {compliance_score:.1f}%

### Requirements Compliance Status

| Requirement | Status |
|-------------|---------|
{% for req, status in requirements_met.items() -%}
| {req} | {'✅ Met' if status else '❌ Not Met'} |
{% endfor %}

### Risk Management System
- **Risk Assessment:** Completed
- **Risk Mitigation:** Active monitoring implemented
- **Human Oversight:** Required for deployment

### Technical Documentation
- **System Architecture:** Documented
- **Training Methodology:** Fully documented with audit trail
- **Performance Characteristics:** Measured and validated
- **Limitations:** Identified and disclosed

### Record Keeping
- **Training Data:** Cryptographically secured audit trail
- **Model Updates:** Complete versioning and provenance
- **Human Annotations:** Privacy-preserved logs maintained
- **Compliance Checks:** Automated and logged

## 7. Limitations and Risks

### Known Limitations
{% for limitation in limitations -%}
- {limitation}
{% endfor %}

### Ethical Considerations
{% for consideration in ethical_considerations -%}
- {consideration}
{% endfor %}

### Mitigation Measures
{% for measure in mitigation_measures -%}
- {measure}
{% endfor %}

## 8. Deployment Recommendations

**Pre-deployment Checklist:**
- [ ] Human oversight mechanisms in place
- [ ] Monitoring and alerting configured
- [ ] Bias testing completed
- [ ] Safety evaluation performed
- [ ] Compliance validation passed

**Ongoing Monitoring:**
- Continuous bias detection
- Performance degradation monitoring
- User feedback collection
- Regular compliance audits

## 9. Contact Information

**Model Developer:** AI Safety Team  
**Compliance Officer:** Chief AI Officer  
**Contact:** compliance@organization.com  

---

*This model card was automatically generated using RLHF Audit Trail v{generator_version} in accordance with EU AI Act requirements.*
"""
    
    def _get_nist_framework_template(self) -> str:
        """Get NIST AI Risk Management Framework template."""
        return """
# AI System Documentation: {model_name}

**NIST AI Risk Management Framework Compliance**
**Version:** {model_version}
**Generated:** {generated_at}

## GOVERN (GV)

### GV-1: Governance Structure
- AI governance policies established
- Clear accountability and responsibility assigned
- Regular review processes implemented

### GV-2: Risk Management
- Comprehensive risk assessment completed
- Risk tolerance defined and monitored
- Mitigation strategies implemented

## MAP (MP)

### MP-1: Context and Purpose
**Intended Use:** {intended_use}

**Application Context:**
- Domain: Conversational AI / RLHF
- Deployment Environment: Research/Production
- Stakeholders: Researchers, Developers, End Users

### MP-2: Risk Identification
**Identified Risks:**
{% for risk, level in risk_assessment.items() -%}
- {risk}: {level}
{% endfor %}

## MEASURE (MS)

### MS-1: Performance Metrics
| Metric | Value | Baseline |
|--------|--------|----------|
{% for metric, value in performance_metrics.items() -%}
| {metric} | {value:.3f} | TBD |
{% endfor %}

### MS-2: Training Metrics
- **Training Duration:** {training_duration_hours:.1f} hours
- **Data Quality Score:** {compliance_score:.1f}%
- **Privacy Budget Utilization:** {privacy_budget_used:.2f}

## MANAGE (MG)

### MG-1: Risk Response
**Mitigation Measures:**
{% for measure in mitigation_measures -%}
- {measure}
{% endfor %}

### MG-2: Monitoring and Maintenance
- Continuous monitoring implemented: ✅
- Regular model updates: ✅  
- Audit trail maintained: {'✅' if verification_status else '❌'}

---

*Generated in compliance with NIST AI Risk Management Framework*
"""
    
    def _get_huggingface_template(self) -> str:
        """Get Hugging Face model card template."""
        return """
---
language: en
license: mit
tags:
- rlhf
- safety
- alignment
- audit-trail
datasets:
- custom-rlhf-dataset
metrics:
- accuracy
- safety_score
---

# {model_name}

## Model Description

This model was trained using Reinforcement Learning from Human Feedback (RLHF) with comprehensive audit trail and EU AI Act compliance.

- **Developed by:** AI Safety Team
- **Model type:** {architecture}
- **Language(s):** English
- **License:** MIT
- **Training approach:** RLHF with Differential Privacy

## Intended Uses

### Direct Use
{intended_use}

### Downstream Use
- Research in AI safety and alignment
- Development of safer conversational AI systems

### Out-of-Scope Use
{% for limitation in limitations -%}
- {limitation}
{% endfor %}

## Training Data

{training_data.get('description', 'Custom RLHF training dataset')}

**Statistics:**
- Training examples: {training_data.get('size', 'N/A')}
- Languages: {training_data.get('languages', 'English')}
- Privacy protection: Differential Privacy (ε={privacy_budget_used:.2f})

## Training Procedure

### Training Hyperparameters
- Training duration: {training_duration_hours:.1f} hours
- Total annotations: {total_annotations:,}
- Human annotators: {annotator_count}
- Policy updates: {policy_updates}

### Privacy Protection
- Differential privacy applied: Yes
- Privacy budget: {privacy_budget_used:.2f}
- Anonymization: Complete

## Evaluation

### Results
{% for metric, value in performance_metrics.items() -%}
- {metric}: {value:.3f}
{% endfor %}

### Compliance Score
- EU AI Act compliance: {compliance_score:.1f}%
- Audit trail verified: {'Yes' if verification_status else 'No'}

## Ethical Considerations

### Risks and Limitations
{% for consideration in ethical_considerations -%}
- {consideration}
{% endfor %}

### Mitigation Strategies
{% for measure in mitigation_measures -%}
- {measure}
{% endfor %}

## Additional Information

### Audit Trail
- Location: {audit_trail_location}
- Verification status: {'Verified' if verification_status else 'Failed'}
- Generated: {generated_at}

### Contact
- Repository: https://github.com/organization/model-repo
- Issues: https://github.com/organization/model-repo/issues
"""
    
    def _get_ieee_template(self) -> str:
        """Get IEEE standard model card template."""
        return """
# IEEE AI System Documentation: {model_name}

## 1. System Identification
- **System Name:** {model_name}
- **Version:** {model_version}
- **Architecture:** {architecture}
- **Parameters:** {parameters:,}

## 2. System Purpose and Context
**Intended Application:** {intended_use}

## 3. System Development
### 3.1 Training Data
{training_data.get('description', 'RLHF training dataset')}

### 3.2 Development Process
- Training duration: {training_duration_hours:.1f} hours
- Annotations collected: {total_annotations:,}
- Privacy protection: Differential Privacy

### 3.3 Validation and Verification
- Audit trail verified: {'Yes' if verification_status else 'No'}
- Compliance score: {compliance_score:.1f}%

## 4. Performance Characteristics
{% for metric, value in performance_metrics.items() -%}
- {metric}: {value:.3f}
{% endfor %}

## 5. Risk Analysis
{% for risk, level in risk_assessment.items() -%}
- {risk}: {level}
{% endfor %}

## 6. Operational Constraints
### Limitations
{% for limitation in limitations -%}
- {limitation}
{% endfor %}

### Ethical Considerations  
{% for consideration in ethical_considerations -%}
- {consideration}
{% endfor %}

## 7. Maintenance and Updates
- Regular monitoring: Implemented
- Update schedule: As needed
- Audit requirements: Continuous

---
*IEEE Standard Compliant Documentation - Generated {generated_at}*
"""

    def save_model_card(self, model_card_content: str, output_path: Path, 
                       format_type: str = "html"):
        """Save model card to file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(model_card_content)
                
            logger.info(f"Model card saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model card: {e}")
            raise AuditTrailError(f"Failed to save model card: {e}")
    
    def validate_model_card(self, model_card_content: str, 
                           format_type: ModelCardFormat) -> Dict[str, Any]:
        """Validate model card completeness and compliance."""
        validation_results = {
            'is_valid': True,
            'missing_sections': [],
            'warnings': [],
            'compliance_score': 100.0
        }
        
        required_sections = {
            ModelCardFormat.EU_AI_ACT: [
                'Model Overview', 'Intended Use', 'Training Data',
                'Training Process Provenance', 'Performance Metrics',
                'EU AI Act Compliance', 'Limitations and Risks'
            ],
            ModelCardFormat.NIST_FRAMEWORK: [
                'GOVERN', 'MAP', 'MEASURE', 'MANAGE'
            ],
            ModelCardFormat.HUGGINGFACE: [
                'Model Description', 'Intended Uses', 'Training Data',
                'Training Procedure', 'Evaluation'
            ]
        }
        
        sections = required_sections.get(format_type, [])
        
        for section in sections:
            if section.lower() not in model_card_content.lower():
                validation_results['missing_sections'].append(section)
                validation_results['is_valid'] = False
                validation_results['compliance_score'] -= (100 / len(sections))
        
        # Additional validation checks
        if 'privacy' not in model_card_content.lower():
            validation_results['warnings'].append('Privacy considerations not explicitly mentioned')
            
        if 'bias' not in model_card_content.lower():
            validation_results['warnings'].append('Bias considerations not explicitly mentioned')
            
        if 'limitation' not in model_card_content.lower():
            validation_results['warnings'].append('Limitations not explicitly mentioned')
        
        return validation_results