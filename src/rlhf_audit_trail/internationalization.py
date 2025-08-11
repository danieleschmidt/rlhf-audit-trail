"""
Internationalization (i18n) and localization (l10n) support for RLHF audit trail.
Global-first implementation with multi-language support and regional compliance.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import locale
import gettext

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    KOREAN = "ko"
    RUSSIAN = "ru"


class Region(Enum):
    """Supported regions for compliance and formatting."""
    # Americas
    US = "US"
    CA = "CA"
    BR = "BR"
    MX = "MX"
    
    # Europe
    EU = "EU"
    GB = "GB"
    DE = "DE"
    FR = "FR"
    IT = "IT"
    ES = "ES"
    NL = "NL"
    
    # Asia Pacific
    JP = "JP"
    CN = "CN"
    IN = "IN"
    AU = "AU"
    KR = "KR"
    SG = "SG"
    
    # Other
    GLOBAL = "GLOBAL"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    language: SupportedLanguage
    region: Region
    timezone: str
    date_format: str
    time_format: str
    number_format: str
    currency_code: str
    decimal_separator: str = "."
    thousands_separator: str = ","
    rtl_support: bool = False  # Right-to-left text support
    
    @classmethod
    def get_default_for_region(cls, region: Region) -> 'LocalizationConfig':
        """Get default localization config for a region."""
        region_defaults = {
            Region.US: cls(
                language=SupportedLanguage.ENGLISH,
                region=Region.US,
                timezone="America/New_York",
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                number_format="1,234.56",
                currency_code="USD"
            ),
            Region.EU: cls(
                language=SupportedLanguage.ENGLISH,
                region=Region.EU,
                timezone="Europe/Brussels",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                currency_code="EUR",
                decimal_separator=",",
                thousands_separator="."
            ),
            Region.DE: cls(
                language=SupportedLanguage.GERMAN,
                region=Region.DE,
                timezone="Europe/Berlin",
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                currency_code="EUR",
                decimal_separator=",",
                thousands_separator="."
            ),
            Region.JP: cls(
                language=SupportedLanguage.JAPANESE,
                region=Region.JP,
                timezone="Asia/Tokyo",
                date_format="%Y/%m/%d",
                time_format="%H:%M",
                number_format="1,234.56",
                currency_code="JPY"
            ),
            Region.CN: cls(
                language=SupportedLanguage.CHINESE_SIMPLIFIED,
                region=Region.CN,
                timezone="Asia/Shanghai",
                date_format="%Y年%m月%d日",
                time_format="%H:%M",
                number_format="1,234.56",
                currency_code="CNY"
            )
        }
        
        return region_defaults.get(region, region_defaults[Region.US])


class TranslationManager:
    """Manages translations and message localization."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize translation manager.
        
        Args:
            base_path: Base path for translation files
        """
        self.base_path = base_path or Path(__file__).parent / "translations"
        self.current_language = SupportedLanguage.ENGLISH
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_language = SupportedLanguage.ENGLISH
        
        self._load_translations()
    
    def _load_translations(self):
        """Load all available translations."""
        # Default translations (embedded for reliability)
        default_translations = {
            SupportedLanguage.ENGLISH.value: {
                # Core messages
                "audit.session.created": "Training session created: {session_id}",
                "audit.session.completed": "Training session completed successfully",
                "audit.annotation.logged": "Annotation batch logged with {count} samples",
                "audit.policy.updated": "Policy updated at step {step}",
                "audit.checkpoint.created": "Checkpoint created for epoch {epoch}",
                "audit.model_card.generated": "Model card generated successfully",
                
                # Error messages
                "error.privacy.budget_exceeded": "Privacy budget exceeded: {details}",
                "error.validation.failed": "Validation failed: {field} - {message}",
                "error.security.threat_detected": "Security threat detected: {threat_type}",
                "error.compliance.violation": "Compliance violation: {framework} - {violation}",
                "error.system.resource_exhausted": "System resources exhausted: {resource}",
                
                # UI labels
                "ui.dashboard.title": "RLHF Audit Trail Dashboard",
                "ui.session.active_sessions": "Active Sessions",
                "ui.privacy.budget_remaining": "Privacy Budget Remaining",
                "ui.compliance.status": "Compliance Status",
                "ui.performance.metrics": "Performance Metrics",
                
                # Compliance messages
                "compliance.eu_ai_act.compliant": "EU AI Act compliance verified",
                "compliance.nist.compliant": "NIST framework compliance verified",
                "compliance.gdpr.data_protected": "Personal data protection verified",
                
                # Status messages
                "status.healthy": "Healthy",
                "status.degraded": "Degraded",
                "status.critical": "Critical",
                "status.offline": "Offline"
            },
            
            SupportedLanguage.SPANISH.value: {
                "audit.session.created": "Sesión de entrenamiento creada: {session_id}",
                "audit.session.completed": "Sesión de entrenamiento completada exitosamente",
                "audit.annotation.logged": "Lote de anotaciones registrado con {count} muestras",
                "audit.policy.updated": "Política actualizada en el paso {step}",
                "audit.checkpoint.created": "Punto de control creado para la época {epoch}",
                
                "error.privacy.budget_exceeded": "Presupuesto de privacidad excedido: {details}",
                "error.validation.failed": "Validación fallida: {field} - {message}",
                "error.security.threat_detected": "Amenaza de seguridad detectada: {threat_type}",
                "error.compliance.violation": "Violación de cumplimiento: {framework} - {violation}",
                
                "ui.dashboard.title": "Panel de Auditoría RLHF",
                "ui.session.active_sessions": "Sesiones Activas",
                "ui.privacy.budget_remaining": "Presupuesto de Privacidad Restante",
                "ui.compliance.status": "Estado de Cumplimiento",
                
                "status.healthy": "Saludable",
                "status.degraded": "Degradado", 
                "status.critical": "Crítico",
                "status.offline": "Desconectado"
            },
            
            SupportedLanguage.FRENCH.value: {
                "audit.session.created": "Session d'entraînement créée: {session_id}",
                "audit.session.completed": "Session d'entraînement terminée avec succès",
                "audit.annotation.logged": "Lot d'annotations enregistré avec {count} échantillons",
                "audit.policy.updated": "Politique mise à jour à l'étape {step}",
                "audit.checkpoint.created": "Point de contrôle créé pour l'époque {epoch}",
                
                "error.privacy.budget_exceeded": "Budget de confidentialité dépassé: {details}",
                "error.validation.failed": "Échec de validation: {field} - {message}",
                "error.security.threat_detected": "Menace de sécurité détectée: {threat_type}",
                "error.compliance.violation": "Violation de conformité: {framework} - {violation}",
                
                "ui.dashboard.title": "Tableau de Bord d'Audit RLHF",
                "ui.session.active_sessions": "Sessions Actives",
                "ui.privacy.budget_remaining": "Budget de Confidentialité Restant",
                "ui.compliance.status": "Statut de Conformité",
                
                "status.healthy": "Sain",
                "status.degraded": "Dégradé",
                "status.critical": "Critique",
                "status.offline": "Hors ligne"
            },
            
            SupportedLanguage.GERMAN.value: {
                "audit.session.created": "Trainingssitzung erstellt: {session_id}",
                "audit.session.completed": "Trainingssitzung erfolgreich abgeschlossen",
                "audit.annotation.logged": "Annotationscharge mit {count} Proben protokolliert",
                "audit.policy.updated": "Richtlinie bei Schritt {step} aktualisiert",
                "audit.checkpoint.created": "Checkpoint für Epoche {epoch} erstellt",
                
                "error.privacy.budget_exceeded": "Datenschutzbudget überschritten: {details}",
                "error.validation.failed": "Validierung fehlgeschlagen: {field} - {message}",
                "error.security.threat_detected": "Sicherheitsbedrohung erkannt: {threat_type}",
                "error.compliance.violation": "Compliance-Verstoß: {framework} - {violation}",
                
                "ui.dashboard.title": "RLHF Audit-Trail Dashboard",
                "ui.session.active_sessions": "Aktive Sitzungen",
                "ui.privacy.budget_remaining": "Verbleibendes Datenschutzbudget",
                "ui.compliance.status": "Compliance-Status",
                
                "status.healthy": "Gesund",
                "status.degraded": "Beeinträchtigt",
                "status.critical": "Kritisch",
                "status.offline": "Offline"
            },
            
            SupportedLanguage.JAPANESE.value: {
                "audit.session.created": "トレーニングセッションが作成されました: {session_id}",
                "audit.session.completed": "トレーニングセッションが正常に完了しました",
                "audit.annotation.logged": "{count}サンプルのアノテーションバッチがログに記録されました",
                "audit.policy.updated": "ステップ{step}でポリシーが更新されました",
                "audit.checkpoint.created": "エポック{epoch}のチェックポイントが作成されました",
                
                "error.privacy.budget_exceeded": "プライバシー予算を超過しました: {details}",
                "error.validation.failed": "検証に失敗しました: {field} - {message}",
                "error.security.threat_detected": "セキュリティ脅威が検出されました: {threat_type}",
                "error.compliance.violation": "コンプライアンス違反: {framework} - {violation}",
                
                "ui.dashboard.title": "RLHF監査証跡ダッシュボード",
                "ui.session.active_sessions": "アクティブセッション",
                "ui.privacy.budget_remaining": "残りのプライバシー予算",
                "ui.compliance.status": "コンプライアンスステータス",
                
                "status.healthy": "正常",
                "status.degraded": "劣化",
                "status.critical": "重要",
                "status.offline": "オフライン"
            },
            
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "audit.session.created": "训练会话已创建：{session_id}",
                "audit.session.completed": "训练会话成功完成",
                "audit.annotation.logged": "已记录包含{count}个样本的注释批次",
                "audit.policy.updated": "策略在步骤{step}处更新",
                "audit.checkpoint.created": "已为轮次{epoch}创建检查点",
                
                "error.privacy.budget_exceeded": "隐私预算超额：{details}",
                "error.validation.failed": "验证失败：{field} - {message}",
                "error.security.threat_detected": "检测到安全威胁：{threat_type}",
                "error.compliance.violation": "合规性违规：{framework} - {violation}",
                
                "ui.dashboard.title": "RLHF审计跟踪仪表板",
                "ui.session.active_sessions": "活跃会话",
                "ui.privacy.budget_remaining": "剩余隐私预算",
                "ui.compliance.status": "合规状态",
                
                "status.healthy": "健康",
                "status.degraded": "降级",
                "status.critical": "严重",
                "status.offline": "离线"
            }
        }
        
        # Load default translations
        for lang_code, messages in default_translations.items():
            self.translations[lang_code] = messages
        
        # Try to load translations from files (if available)
        try:
            if self.base_path.exists():
                for lang_file in self.base_path.glob("*.json"):
                    lang_code = lang_file.stem
                    try:
                        with open(lang_file, 'r', encoding='utf-8') as f:
                            file_translations = json.load(f)
                            if lang_code in self.translations:
                                self.translations[lang_code].update(file_translations)
                            else:
                                self.translations[lang_code] = file_translations
                        logger.debug(f"Loaded translations from {lang_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load translations from {lang_file}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load translation files: {e}")
    
    def set_language(self, language: Union[SupportedLanguage, str]):
        """Set current language for translations."""
        if isinstance(language, str):
            try:
                language = SupportedLanguage(language)
            except ValueError:
                logger.warning(f"Unsupported language: {language}, falling back to English")
                language = SupportedLanguage.ENGLISH
        
        self.current_language = language
        logger.info(f"Language set to: {language.value}")
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Translate a message key to current language.
        
        Args:
            key: Translation key
            **kwargs: Format parameters for the message
            
        Returns:
            Translated and formatted message
        """
        lang_code = self.current_language.value
        
        # Try current language
        if lang_code in self.translations and key in self.translations[lang_code]:
            message = self.translations[lang_code][key]
        # Fall back to English
        elif self.fallback_language.value in self.translations:
            fallback_lang = self.fallback_language.value
            if key in self.translations[fallback_lang]:
                message = self.translations[fallback_lang][key]
                logger.debug(f"Used fallback translation for key: {key}")
            else:
                message = key  # Return key if no translation found
                logger.warning(f"No translation found for key: {key}")
        else:
            message = key
        
        # Format message with parameters
        try:
            return message.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing format parameter {e} for key: {key}")
            return message
        except Exception as e:
            logger.warning(f"Translation formatting error for key {key}: {e}")
            return message
    
    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        return list(self.translations.keys())
    
    def add_translation(self, language: str, key: str, message: str):
        """Add or update a translation."""
        if language not in self.translations:
            self.translations[language] = {}
        self.translations[language][key] = message


class DateTimeFormatter:
    """Handles date/time formatting for different regions."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
    
    def format_datetime(self, dt: datetime, include_timezone: bool = True) -> str:
        """Format datetime according to regional preferences."""
        try:
            # Convert to local timezone if needed
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Format date and time
            date_str = dt.strftime(self.config.date_format)
            time_str = dt.strftime(self.config.time_format)
            
            formatted = f"{date_str} {time_str}"
            
            if include_timezone:
                tz_name = self.config.timezone.split('/')[-1].replace('_', ' ')
                formatted += f" ({tz_name})"
            
            return formatted
            
        except Exception as e:
            logger.warning(f"DateTime formatting error: {e}")
            return dt.isoformat()
    
    def format_date(self, dt: datetime) -> str:
        """Format date only."""
        try:
            return dt.strftime(self.config.date_format)
        except Exception as e:
            logger.warning(f"Date formatting error: {e}")
            return dt.strftime("%Y-%m-%d")
    
    def format_time(self, dt: datetime) -> str:
        """Format time only."""
        try:
            return dt.strftime(self.config.time_format)
        except Exception as e:
            logger.warning(f"Time formatting error: {e}")
            return dt.strftime("%H:%M")


class NumberFormatter:
    """Handles number formatting for different regions."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
    
    def format_number(self, number: Union[int, float], precision: int = 2) -> str:
        """Format number according to regional preferences."""
        try:
            if isinstance(number, int):
                # Format integer
                formatted = f"{number:,}"
            else:
                # Format float with precision
                formatted = f"{number:,.{precision}f}"
            
            # Apply regional separators
            if self.config.decimal_separator != "." or self.config.thousands_separator != ",":
                formatted = formatted.replace(",", "TEMP")
                formatted = formatted.replace(".", self.config.decimal_separator)
                formatted = formatted.replace("TEMP", self.config.thousands_separator)
            
            return formatted
            
        except Exception as e:
            logger.warning(f"Number formatting error: {e}")
            return str(number)
    
    def format_currency(self, amount: float, precision: int = 2) -> str:
        """Format currency amount."""
        try:
            formatted_number = self.format_number(amount, precision)
            
            # Currency symbol placement varies by region
            currency_symbols = {
                "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", 
                "CNY": "¥", "KRW": "₩", "CAD": "C$", "AUD": "A$"
            }
            
            symbol = currency_symbols.get(self.config.currency_code, self.config.currency_code)
            
            # Most regions put symbol before amount
            if self.config.currency_code in ["EUR"] and self.config.region == Region.EU:
                return f"{formatted_number} {symbol}"
            else:
                return f"{symbol}{formatted_number}"
                
        except Exception as e:
            logger.warning(f"Currency formatting error: {e}")
            return f"{self.config.currency_code} {amount}"
    
    def format_percentage(self, value: float, precision: int = 1) -> str:
        """Format percentage value."""
        try:
            formatted = self.format_number(value, precision)
            return f"{formatted}%"
        except Exception as e:
            logger.warning(f"Percentage formatting error: {e}")
            return f"{value}%"


class ComplianceLocalizer:
    """Handles compliance-specific localization requirements."""
    
    def __init__(self, region: Region):
        self.region = region
        self.compliance_frameworks = self._get_regional_frameworks()
    
    def _get_regional_frameworks(self) -> List[str]:
        """Get applicable compliance frameworks for region."""
        regional_frameworks = {
            Region.EU: ["EU_AI_ACT", "GDPR", "ISO_27001"],
            Region.US: ["NIST_AI_RMF", "SOC2", "FedRAMP"],
            Region.CA: ["PIPEDA", "AIDA"],
            Region.GB: ["UK_GDPR", "AI_WHITE_PAPER"],
            Region.CN: ["PIPL", "CYBERSECURITY_LAW"],
            Region.JP: ["PERSONAL_INFO_PROTECTION_ACT"],
            Region.AU: ["PRIVACY_ACT", "AI_ETHICS_PRINCIPLES"],
            Region.BR: ["LGPD"],
            Region.IN: ["DPDP_ACT"],
            Region.KR: ["PIPA"],
            Region.SG: ["PDPA", "AI_GOVERNANCE"]
        }
        
        return regional_frameworks.get(self.region, ["GLOBAL_BEST_PRACTICES"])
    
    def get_compliance_requirements(self) -> Dict[str, Any]:
        """Get region-specific compliance requirements."""
        requirements = {
            "frameworks": self.compliance_frameworks,
            "data_residency": self._get_data_residency_requirements(),
            "audit_retention": self._get_audit_retention_requirements(),
            "privacy_rights": self._get_privacy_rights(),
            "reporting_requirements": self._get_reporting_requirements()
        }
        
        return requirements
    
    def _get_data_residency_requirements(self) -> Dict[str, Any]:
        """Get data residency requirements."""
        residency_rules = {
            Region.EU: {
                "data_must_stay_in_region": True,
                "allowed_transfers": ["ADEQUACY_DECISION", "STANDARD_CONTRACTUAL_CLAUSES"],
                "restricted_countries": ["US", "CN", "RU"]
            },
            Region.CN: {
                "data_must_stay_in_region": True,
                "cross_border_assessment_required": True,
                "restricted_transfers": True
            },
            Region.RU: {
                "data_must_stay_in_region": True,
                "local_storage_required": True
            },
            Region.US: {
                "data_can_transfer": True,
                "sector_specific_rules": ["HEALTHCARE_HIPAA", "FINANCIAL_SOX"]
            }
        }
        
        return residency_rules.get(self.region, {"flexible_transfer": True})
    
    def _get_audit_retention_requirements(self) -> Dict[str, Any]:
        """Get audit log retention requirements."""
        retention_rules = {
            Region.EU: {"min_years": 3, "max_years": 7, "deletion_right": True},
            Region.US: {"min_years": 7, "sector_specific": True},
            Region.CN: {"min_years": 3, "government_access": True},
            Region.JP: {"min_years": 5, "anonymization_preferred": True},
            Region.AU: {"min_years": 7, "privacy_act_compliance": True},
            Region.CA: {"min_years": 3, "pipeda_compliance": True}
        }
        
        return retention_rules.get(self.region, {"min_years": 3})
    
    def _get_privacy_rights(self) -> List[str]:
        """Get privacy rights applicable in region."""
        privacy_rights = {
            Region.EU: [
                "RIGHT_TO_ACCESS", "RIGHT_TO_RECTIFICATION", "RIGHT_TO_ERASURE",
                "RIGHT_TO_PORTABILITY", "RIGHT_TO_OBJECT", "RIGHT_TO_RESTRICT"
            ],
            Region.US: ["NOTICE", "CHOICE", "ACCESS", "SECURITY"],
            Region.CA: ["ACCESS", "CORRECTION", "WITHDRAWAL", "PORTABILITY"],
            Region.CN: ["INFORMED_CONSENT", "DELETION", "PORTABILITY"],
            Region.AU: ["ACCESS", "CORRECTION", "ERASURE"],
            Region.BR: ["ACCESS", "RECTIFICATION", "DELETION", "PORTABILITY"]
        }
        
        return privacy_rights.get(self.region, ["ACCESS", "CORRECTION"])
    
    def _get_reporting_requirements(self) -> Dict[str, Any]:
        """Get regulatory reporting requirements."""
        reporting_rules = {
            Region.EU: {
                "breach_notification_hours": 72,
                "dpa_reporting_required": True,
                "ai_act_conformity_required": True
            },
            Region.US: {
                "sector_specific_reporting": True,
                "ftc_enforcement": True
            },
            Region.CN: {
                "cyberspace_administration_reporting": True,
                "algorithm_registration_required": True
            }
        }
        
        return reporting_rules.get(self.region, {})


class InternationalizationManager:
    """Main internationalization and localization manager."""
    
    def __init__(self, language: SupportedLanguage = SupportedLanguage.ENGLISH, 
                 region: Region = Region.GLOBAL):
        """
        Initialize internationalization manager.
        
        Args:
            language: Default language
            region: Default region
        """
        self.language = language
        self.region = region
        self.config = LocalizationConfig.get_default_for_region(region)
        
        # Initialize components
        self.translator = TranslationManager()
        self.translator.set_language(language)
        
        self.datetime_formatter = DateTimeFormatter(self.config)
        self.number_formatter = NumberFormatter(self.config)
        self.compliance_localizer = ComplianceLocalizer(region)
        
        logger.info(f"Internationalization initialized: {language.value} / {region.value}")
    
    def set_locale(self, language: SupportedLanguage, region: Region):
        """Set locale (language and region)."""
        self.language = language
        self.region = region
        self.config = LocalizationConfig.get_default_for_region(region)
        
        # Update components
        self.translator.set_language(language)
        self.datetime_formatter = DateTimeFormatter(self.config)
        self.number_formatter = NumberFormatter(self.config)
        self.compliance_localizer = ComplianceLocalizer(region)
        
        logger.info(f"Locale changed to: {language.value} / {region.value}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate message with current locale."""
        return self.translator.translate(key, **kwargs)
    
    def format_datetime(self, dt: datetime, include_timezone: bool = True) -> str:
        """Format datetime for current locale."""
        return self.datetime_formatter.format_datetime(dt, include_timezone)
    
    def format_number(self, number: Union[int, float], precision: int = 2) -> str:
        """Format number for current locale."""
        return self.number_formatter.format_number(number, precision)
    
    def format_currency(self, amount: float, precision: int = 2) -> str:
        """Format currency for current locale."""
        return self.number_formatter.format_currency(amount, precision)
    
    def get_compliance_info(self) -> Dict[str, Any]:
        """Get compliance information for current region."""
        return self.compliance_localizer.get_compliance_requirements()
    
    def localize_audit_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Localize audit event data."""
        localized_event = event_data.copy()
        
        # Translate event type
        event_type = event_data.get('event_type', '')
        if event_type:
            localized_event['event_type_display'] = self.translate(f"audit.{event_type}")
        
        # Format timestamps
        if 'timestamp' in event_data:
            try:
                if isinstance(event_data['timestamp'], str):
                    dt = datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00'))
                else:
                    dt = event_data['timestamp']
                localized_event['timestamp_formatted'] = self.format_datetime(dt)
            except Exception as e:
                logger.warning(f"Failed to format timestamp: {e}")
        
        # Format numeric values
        for key, value in event_data.items():
            if isinstance(value, (int, float)) and not key.endswith('_id'):
                if 'budget' in key or 'epsilon' in key:
                    localized_event[f"{key}_formatted"] = self.format_number(value, 4)
                elif 'percent' in key or 'rate' in key:
                    localized_event[f"{key}_formatted"] = self.number_formatter.format_percentage(value)
                else:
                    localized_event[f"{key}_formatted"] = self.format_number(value)
        
        return localized_event
    
    def get_locale_info(self) -> Dict[str, Any]:
        """Get current locale information."""
        return {
            "language": self.language.value,
            "region": self.region.value,
            "timezone": self.config.timezone,
            "currency": self.config.currency_code,
            "date_format": self.config.date_format,
            "time_format": self.config.time_format,
            "rtl_support": self.config.rtl_support,
            "decimal_separator": self.config.decimal_separator,
            "thousands_separator": self.config.thousands_separator
        }


# Global internationalization manager
_global_i18n_manager: Optional[InternationalizationManager] = None

def get_i18n_manager() -> InternationalizationManager:
    """Get global internationalization manager."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager()
    return _global_i18n_manager

def set_global_locale(language: Union[SupportedLanguage, str], region: Union[Region, str]):
    """Set global locale."""
    if isinstance(language, str):
        language = SupportedLanguage(language)
    if isinstance(region, str):
        region = Region(region)
    
    manager = get_i18n_manager()
    manager.set_locale(language, region)

def translate(key: str, **kwargs) -> str:
    """Convenient translation function."""
    return get_i18n_manager().translate(key, **kwargs)

def format_datetime_locale(dt: datetime, include_timezone: bool = True) -> str:
    """Convenient datetime formatting function."""
    return get_i18n_manager().format_datetime(dt, include_timezone)

def format_number_locale(number: Union[int, float], precision: int = 2) -> str:
    """Convenient number formatting function."""
    return get_i18n_manager().format_number(number, precision)