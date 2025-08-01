"""
LLM-based certifications extraction from resume text.
Uses OpenAI/LangChain with function calling for reliable extraction of professional certifications,
licenses, and credentials. Excludes formal education degrees.
"""
import logging
import time
import re
from typing import Dict, List, Optional, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

from src.models.resume import Certification
from src.services.openai_service import langgraph_openai_service, OpenAIServiceError
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificationsExtractionError(Exception):
    """Custom exception for certifications extraction errors."""
    pass


class CertificationsExtractionResult(BaseModel):
    """Certifications extraction result with metadata."""
    certifications: List[Certification] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_active_certifications(self) -> List[Certification]:
        """Get certifications that are still active (not expired)."""
        active = []
        for cert in self.certifications:
            if not cert.expiry or cert.expiry.lower() in ['never', 'lifetime', 'permanent', 'no expiry']:
                active.append(cert)
            # Could add date comparison logic here for expiry dates
        return active
    
    def get_certifications_by_issuer(self, issuer: str) -> List[Certification]:
        """Get all certifications from a specific issuer."""
        issuer_lower = issuer.lower()
        return [cert for cert in self.certifications 
                if issuer_lower in cert.issuer.lower()]
    
    def get_cloud_certifications(self) -> List[Certification]:
        """Get cloud-related certifications."""
        cloud_keywords = ['aws', 'azure', 'google cloud', 'gcp', 'cloud']
        cloud_certs = []
        for cert in self.certifications:
            cert_text = f"{cert.name} {cert.issuer}".lower()
            if any(keyword in cert_text for keyword in cloud_keywords):
                cloud_certs.append(cert)
        return cloud_certs
    
    def get_recent_certifications(self, years: int = 3) -> List[Certification]:
        """Get certifications obtained within the last N years."""
        from datetime import datetime, timedelta
        cutoff_year = datetime.now().year - years
        
        recent = []
        for cert in self.certifications:
            if cert.date:
                try:
                    # Extract year from date
                    year_match = re.search(r'\b(20\d{2})\b', cert.date)
                    if year_match and int(year_match.group()) >= cutoff_year:
                        recent.append(cert)
                except:
                    continue
        return recent
    
    def get_issuers(self) -> List[str]:
        """Get list of unique certification issuers."""
        return list(set(cert.issuer for cert in self.certifications))


class CertificationsLLMResult(BaseModel):
    """Wrapper for LLM extraction result."""
    certifications: List[Certification] = Field(default_factory=list)


class LLMCertificationsExtractor:
    """LLM-based certifications extraction system."""
    
    def __init__(self):
        """Initialize the certifications extractor."""
        self.openai_service = langgraph_openai_service
        logger.info("LLM Certifications Extractor initialized")
    
    async def extract_certifications_from_text(self, text: str) -> CertificationsExtractionResult:
        """
        Extract certifications from resume text using LLM.
        
        Args:
            text: The resume text to extract certifications from
        
        Returns:
            CertificationsExtractionResult with extracted certifications data and metadata
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LLM certifications extraction")
            
            # Perform LLM extraction
            certifications_list = await self._perform_llm_extraction(text)
            
            # Post-process certifications
            processed_certifications = self._post_process_certifications(certifications_list)
            
            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = {
                "processing_time": processing_time,
                "total_certifications": len(processed_certifications),
                "issuers": list(set(cert.issuer for cert in processed_certifications)),
                "active_certifications": len(self._get_active_certs(processed_certifications)),
                "cloud_certifications": len(self._get_cloud_certs(processed_certifications)),
                "model_used": settings.openai_model,
                "extraction_method": "llm_function_calling"
            }
            
            result = CertificationsExtractionResult(
                certifications=processed_certifications,
                extraction_metadata=metadata
            )
            
            logger.info(f"Certifications extraction completed. Found {len(processed_certifications)} certifications in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Certifications extraction failed: {e}")
            raise CertificationsExtractionError(f"Failed to extract certifications: {e}")
    
    async def _perform_llm_extraction(self, text: str) -> List[Certification]:
        """Perform the actual LLM-based certifications extraction."""
        
        # Use function calling for reliable extraction
        structured_llm = self.openai_service.llm.with_structured_output(
            CertificationsLLMResult,
            method="function_calling"
        )
        
        # Create specialized certifications extraction prompt
        system_prompt = """You are an expert at extracting professional certifications from resumes.
        Extract ONLY professional certifications, licenses, and credentials - NOT formal education degrees.

        EXTRACT ONLY:
        - Professional certifications (AWS, Microsoft, Google, Cisco, etc.)
        - Industry certifications (PMP, CISSP, Six Sigma, etc.)
        - Professional licenses (CPA, PE, Bar admission, etc.)
        - Technical certifications (CompTIA, Oracle, Salesforce, etc.)
        - Trade certifications and professional credentials
        - Online course completion certificates (Coursera, Udemy with certificates)

        DO NOT EXTRACT:
        - University degrees (Bachelor's, Master's, PhD)
        - College diplomas or academic programs
        - School-based education or bootcamps
        - Academic coursework or classes

        EXTRACTION GUIDELINES:
        1. Extract each certification as a separate entry
        2. Focus on professional and industry credentials
        3. Extract dates exactly as shown in the resume
        4. Include issuing organization/company
        5. Extract credential IDs, license numbers when mentioned
        6. Include expiry dates when available
    

        WHAT TO EXTRACT FOR EACH CERTIFICATION:
        - Certification name (full official name)
        - Issuing organization/company
        - Date obtained (preserve original format)
        - Expiry date (if mentioned, or "Never" for lifetime certs)
        - Credential ID or license number (if provided)
    

        COMMON CERTIFICATION TYPES:
        Cloud: AWS Certified Solutions Architect, Microsoft Azure, Google Cloud
        Project Management: PMP, Scrum Master, Agile
        Security: CISSP, CompTIA Security+, CEH
        IT: CompTIA A+, Network+, CCNA, MCSE
        Quality: Six Sigma, Lean
        Professional: CPA, PE, Bar License
        Sales/Marketing: Salesforce, HubSpot, Google Ads

        DATE EXTRACTION RULES:
        - Extract dates exactly as shown in the resume
        - Common formats: "2021", "Jan 2021", "01/2021", "Valid until 2024"
        - Use "Never" for lifetime certifications
        - Use "Not specified" if no date mentioned

        ISSUER STANDARDIZATION:
        - Use official organization names: "Amazon Web Services" not "AWS"
        - "Microsoft" not "MS", "Google" not "GOOG"
        - "Project Management Institute" for PMP
        - Keep original if uncertain"""

        human_prompt = """Extract ONLY professional certifications and credentials from this resume text:

        RESUME TEXT:
        {text}

        Look for certifications in sections like:
        - "Certifications" / "Certificates" / "Professional Certifications"
        - "Licenses" / "Credentials" / "Professional Licenses"
        - "Technical Certifications" / "Industry Certifications"

        IMPORTANT: Extract ONLY professional certifications and credentials:
        ✅ INCLUDE:
        - Professional certifications (AWS, Microsoft, Google, Cisco)
        - Industry certifications (PMP, CISSP, Six Sigma)
        - Professional licenses (CPA, PE, Bar License)
        - Technical certifications (CompTIA, Oracle, Salesforce)
        - Trade certifications and professional credentials
        - Online course certificates with professional value

        ❌ DO NOT INCLUDE:
        - University degrees (Bachelor's, Master's, PhD)
        - College diplomas or academic programs
        - School-based education or academic courses
        - Bootcamps or academic training programs

        For each CERTIFICATION found, extract:
        1. **Certification Name**: Full official name of the certification
        2. **Issuing Organization**: Company or organization that issued it
        3. **Date Obtained**: When the certification was earned
        4. **Expiry Date**: When it expires (or "Never" for lifetime)
        5. **Credential Details**: ID numbers  if mentioned

        Extract dates exactly as written in the resume. Do not modify the format.

        If any information is missing or unclear, use "Not specified" for that field.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"text": text})
        
        return result.certifications
    
    def _post_process_certifications(self, certifications_list: List[Certification]) -> List[Certification]:
        """Post-process and validate extracted certifications."""
        
        processed = []
        
        for cert in certifications_list:
            try:
                # Clean and validate the certification entry
                cleaned_cert = self._clean_certification_entry(cert)
                
                # Skip entries with missing critical information
                if not cleaned_cert.name or not cleaned_cert.issuer:
                    logger.warning(f"Skipping incomplete certification: {cleaned_cert.name} from {cleaned_cert.issuer}")
                    continue
                
                # Skip very generic or unclear entries
                if len(cleaned_cert.name.strip()) < 3 or len(cleaned_cert.issuer.strip()) < 2:
                    logger.warning(f"Skipping unclear certification: {cleaned_cert.name}")
                    continue
                
                # Filter out academic degrees that might have been extracted
                if self._is_academic_degree(cleaned_cert.name):
                    logger.warning(f"Filtering out academic degree: {cleaned_cert.name}")
                    continue
                
                processed.append(cleaned_cert)
                
            except Exception as e:
                logger.warning(f"Error processing certification entry: {e}")
                continue
        
        # Sort by date (most recent first)
        processed.sort(
            key=lambda x: self._parse_certification_date_for_sorting(x.date),
            reverse=True
        )
        
        return processed
    
    def _clean_certification_entry(self, cert: Certification) -> Certification:
        """Clean and standardize a single certification entry."""
        
        # Clean certification name
        name = cert.name.strip() if cert.name else ""
        name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
        name = self._standardize_certification_name(name)
        
        # Clean issuer name
        issuer = cert.issuer.strip() if cert.issuer else ""
        issuer = re.sub(r'\s+', ' ', issuer)  # Normalize whitespace
        issuer = self._standardize_issuer_name(issuer)
        
        # Clean date (preserve format)
        date = self._clean_date(cert.date)
        
        # Clean expiry date
        expiry = self._clean_date(cert.expiry)
        
        # Clean credential ID
        credential_id = cert.credential_id.strip() if cert.credential_id else None
        if credential_id and credential_id.lower() in ['not specified', 'n/a', 'na', 'none']:
            credential_id = None
        
        
        return Certification(
            name=name,
            issuer=issuer,
            date=date,
            expiry=expiry,
            credential_id=credential_id
        )
    
    def _standardize_certification_name(self, name: str) -> str:
        """Standardize certification names for consistency."""
        if not name:
            return name
        
        # Common certification name standardizations
        standardizations = {
            'aws certified solutions architect': 'AWS Certified Solutions Architect',
            'aws solutions architect': 'AWS Certified Solutions Architect',
            'certified scrum master': 'Certified ScrumMaster',
            'scrum master': 'Certified ScrumMaster',
            'pmp': 'Project Management Professional (PMP)',
            'cissp': 'Certified Information Systems Security Professional (CISSP)',
            'comptia security+': 'CompTIA Security+',
            'security+': 'CompTIA Security+',
            'comptia a+': 'CompTIA A+',
            'a+': 'CompTIA A+',
            'six sigma black belt': 'Six Sigma Black Belt',
            'six sigma green belt': 'Six Sigma Green Belt',
            'cpa': 'Certified Public Accountant (CPA)',
            'pe': 'Professional Engineer (PE)'
        }
        
        name_lower = name.lower()
        for key, standard in standardizations.items():
            if key in name_lower:
                return standard
        
        # Return title case if no standardization found
        return name.title()
    
    def _standardize_issuer_name(self, issuer: str) -> str:
        """Standardize issuer names for consistency."""
        if not issuer:
            return issuer
        
        # Common issuer standardizations
        standardizations = {
            'aws': 'Amazon Web Services',
            'amazon': 'Amazon Web Services',
            'microsoft': 'Microsoft',
            'ms': 'Microsoft',
            'google': 'Google',
            'gcp': 'Google Cloud',
            'google cloud': 'Google Cloud',
            'cisco': 'Cisco Systems',
            'comptia': 'CompTIA',
            'pmi': 'Project Management Institute',
            'scrum alliance': 'Scrum Alliance',
            'scrumalliance': 'Scrum Alliance',
            'isc2': 'ISC2',
            '(isc)²': 'ISC2',
            'oracle': 'Oracle Corporation',
            'salesforce': 'Salesforce',
            'vmware': 'VMware'
        }
        
        issuer_lower = issuer.lower()
        for key, standard in standardizations.items():
            if key == issuer_lower or key in issuer_lower:
                return standard
        
        # Return title case if no standardization found
        return issuer.title()
    
    def _clean_date(self, date_str: Optional[str]) -> Optional[str]:
        """Clean certification date while preserving format."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Handle common variations
        if date_str.lower() in ['not specified', 'n/a', 'na', 'none', 'tbd', 'pending']:
            return None
        
        # Handle "Never" expiry
        if date_str.lower() in ['never', 'lifetime', 'permanent', 'no expiry', 'does not expire']:
            return "Never"
        
        return date_str
    
    def _is_academic_degree(self, name: str) -> bool:
        """Check if a certification name is actually an academic degree."""
        name_lower = name.lower()
        
        academic_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'diploma',
            'degree', 'university', 'college', 'school',
            'b.s', 'b.a', 'm.s', 'm.a', 'mba', 'ph.d'
        ]
        
        return any(keyword in name_lower for keyword in academic_keywords)
    
    def _parse_certification_date_for_sorting(self, date_str: Optional[str]) -> int:
        """Parse certification date for sorting purposes."""
        if not date_str:
            return 0
        
        # Extract year from various formats
        try:
            # Look for 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                return int(year_match.group())
            
            # Try direct conversion
            return int(date_str)
        except (ValueError, TypeError):
            return 0
    
    def _get_active_certs(self, certifications: List[Certification]) -> List[Certification]:
        """Helper method to get active certifications for metadata."""
        result = CertificationsExtractionResult(certifications=certifications)
        return result.get_active_certifications()
    
    def _get_cloud_certs(self, certifications: List[Certification]) -> List[Certification]:
        """Helper method to get cloud certifications for metadata."""
        result = CertificationsExtractionResult(certifications=certifications)
        return result.get_cloud_certifications()


# Global extractor instance
llm_certifications_extractor = LLMCertificationsExtractor()


# Convenience functions
async def extract_certifications_from_resume(resume_text: str) -> CertificationsExtractionResult:
    """Extract certifications from resume text."""
    return await llm_certifications_extractor.extract_certifications_from_text(resume_text)


async def extract_certifications_list(resume_text: str) -> List[Certification]:
    """Extract certifications and return just the list of Certification objects."""
    result = await extract_certifications_from_resume(resume_text)
    return result.certifications


if __name__ == "__main__":
    import asyncio
    # Test with real file if available
    try:
        with open("/home/user/ranalyser/tests/data/sample_resumes/senior_developer.txt", "r") as file:
            resume_text = file.read()
        
        print("=" * 60)
        print("Testing with real resume file...")
        result = asyncio.run(extract_certifications_from_resume(resume_text))
        
        print("✅ Real file test successful!")
        print(f"Found {len(result.certifications)} certifications")
        print(f"Processing time: {result.extraction_metadata['processing_time']:.2f}s")
        
        for i, cert in enumerate(result.certifications, 1):
            print(f"{i}. {cert.name}")
            print(f"   Issuer: {cert.issuer} ({cert.date})")
            
    except FileNotFoundError:
        print("Sample resume file not found, skipping file test")
    except Exception as e:
        print(f"Real file test error: {e}")