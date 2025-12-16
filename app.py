import os
import json
import time
import tempfile
import logging
import re
from contextlib import contextmanager
from typing import List, Dict, Optional
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_from_directory
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Choose best PDF processing library
try:
    import fitz  # PyMuPDF - much better than pdf2image
    PDF_METHOD = 'pymupdf'
except ImportError:
    try:
        from pdf2image import convert_from_path
        PDF_METHOD = 'pdf2image'
    except ImportError:
        PDF_METHOD = 'direct'  # Use Gemini's native PDF support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_PIXELS = 50_000_000  # Better than unlimited
MAX_DIMENSION = 2000
MIN_DIMENSION = 800

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# === STRUCTURED OUTPUT: Pydantic schema for Gemini field extraction ===
class FieldItem(BaseModel):
    name: str = Field(description="Name of the extracted field.")
    value: str = Field(description="Value of the extracted field.")

class BOMItem(BaseModel):
    part_code: str = Field(description="Part code or number.")
    description: str = Field(description="Description of the part.")
    material: str = Field(description="Material of the part.")
    quantity: str = Field(description="Quantity of the part.")
    unit_price: str = Field(description="Unit price, if available.")
    total_price: str = Field(description="Total price, if available.")
    remarks: str = Field(description="Remarks or notes.")

class FieldsModel(BaseModel):
    fields: List[FieldItem]

class BOMModel(BaseModel):
    bom_items: List[BOMItem]

# Initialize Gemini client
gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if gemini_api_key:
    client = genai.Client(api_key=gemini_api_key)
    
    # Configuration for field extraction (fast Flash model)
    generation_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        max_output_tokens=4096,  # Increased for detailed extractions
        response_mime_type="application/json",
    )

    # Configuration for BOM extraction
    bom_generation_config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,  # Even more for BOM tables
        response_mime_type="application/json",
    )

    # Model names
    model = 'gemini-2.5-flash'
    bom_model = 'gemini-2.5-flash'
else:
    client = None
    model = None
    bom_model = None
    logger.warning("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('EMAIL_USER')

mail = Mail(app)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@contextmanager
def safe_image_context(image_path):
    """Context manager for safe image handling with proper cleanup"""
    img = None
    try:
        # Set reasonable limit instead of unlimited
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(image_path)
        yield img
    finally:
        if img:
            img.close()

def extract_tables_with_pymupdf(pdf_path):
    """Extract table-like structures using PyMuPDF only (no Poppler)."""
    tables = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            rows = []
            for b in blocks:
                text = b[4].strip()
                if not text:
                    continue
                # split into columns
                cols = [c.strip() for c in text.split() if c.strip()]
                if len(cols) > 1:
                    rows.append(cols)
            if rows:
                logger.info(f"Page {page_num}: extracted {len(rows)} potential rows")
                tables.append({"page": page_num, "data": rows})
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF table extraction failed: {e}")
    return tables

def extract_bom_with_tables(file_path, chunk_size=50):
    if not bom_model or not client:
        return {'bom_items': []}

    tables = extract_tables_with_pymupdf(file_path)
    if not tables:
        return {'bom_items': []}

    all_bom_items = []
    raw_outputs = []

    for t in tables:
        rows = t["data"]
        logger.info(f"Processing page {t['page']} with {len(rows)} rows...")

        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i+chunk_size]
            chunk_text = "\n".join([", ".join(r) for r in chunk])

            prompt = f"""
                        You are a CAD/BOM extraction system.
                        Interpret the following rows and return ONLY valid JSON.
                        IMPORTANT INSTRUCTIONS:
                        - Look for quantity information in ANY column (numbers like 1, 2, 10, etc.)
                        - If no explicit quantity column exists, extract quantities from part codes, descriptions, or other fields
                        - Look for patterns like "N.4", "QTY: 2", "x3", "2pcs", etc.
                        - If no quantity is found, use "1" as default
                        - Extract unit prices if available (look for currency symbols, decimal numbers)
                        - Calculate total_price = quantity × unit_price when possible
                        - Be thorough in extracting part codes and descriptions
                        If not a BOM, return {{"bom_items": []}}.
                        Rows (page {t['page']}, chunk {i//chunk_size+1}):
                        {chunk_text}
                        """

            try:
                # Use structured output with Pydantic model
                response = client.models.generate_content(
                    model=bom_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        response_schema=BOMModel,  # Pass Pydantic model directly
                    ),
                )
                
                # Parse structured response
                bom_result = BOMModel.model_validate_json(response.text)
                items = [item.model_dump() for item in bom_result.bom_items]
                raw_output = response.text
                logger.info(f"BOM chunk processed: {len(items)} items (structured)")
                
            except Exception as e:
                logger.warning(f"Structured BOM output failed: {e}. Falling back to text parsing.")
                # Fallback to text parsing
                try:
                    response = client.models.generate_content(
                        model=bom_model,
                        contents=prompt,
                        config=bom_generation_config,
                    )
                    raw_output = response.text
                    parsed = parse_bom_response(raw_output)
                    items = parsed.get("bom_items", [])
                    logger.info(f"BOM chunk processed: {len(items)} items (fallback)")
                except Exception as e2:
                    logger.error(f"BOM extraction failed completely: {e2}")
                    items = []
                    raw_output = ""

            raw_outputs.append(raw_output)

            if not items:
                logger.warning(f"BOM chunk {i//chunk_size+1} on page {t['page']} returned empty")
                continue

            # Save to CSV incrementally
            chunk_df = pd.DataFrame(items)
            if not chunk_df.empty:
                out_csv = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.basename(file_path)}_bom.csv")
                if os.path.exists(out_csv):
                    chunk_df.to_csv(out_csv, mode='a', header=False, index=False)
                else:
                    chunk_df.to_csv(out_csv, index=False)

            all_bom_items.extend(items)

    return {
        'bom_items': all_bom_items,
        'bom_raw': "\n\n---\n\n".join(raw_outputs)
    }

def parse_bom_response(response_text):
    """Fallback parser for BOM responses"""
    try:
        response_text = response_text.strip()
        result = json.loads(response_text)

        if isinstance(result, dict) and 'bom_items' in result:
            # Post-process to ensure quantity is always present
            for item in result['bom_items']:
                if 'quantity' not in item or not item['quantity'] or item['quantity'].strip() == '':
                    quantity = extract_quantity_from_text(
                        item.get('part_code', '') + ' ' + item.get('description', '')
                    )
                    item['quantity'] = quantity if quantity else '1'
                
                # Ensure all required fields exist
                item.setdefault('unit_price', '')
                item.setdefault('total_price', '')
                item.setdefault('remarks', item.get('remarks', ''))
            
            return result
        elif isinstance(result, list):
            return {'bom_items': result}
    except Exception as e:
        logger.error(f"BOM JSON parsing failed: {e}")

    return {'bom_items': [], 'bom_raw': response_text}

def extract_quantity_from_text(text):
    """Extract quantity from text using regex patterns"""
    if not text:
        return None
    
    # Common quantity patterns
    patterns = [
        r'N\.(\d+)',           # N.4
        r'QTY[:\s]*(\d+)',     # QTY: 2, QTY 3
        r'x(\d+)',             # x3
        r'(\d+)pcs',           # 2pcs
        r'(\d+)\s*pieces',     # 3 pieces
        r'(\d+)\s*units',      # 5 units
        r'^(\d+)$',            # Just a number
        r'(\d+)\s*ea',         # 2 ea (each)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def convert_pdf_with_pymupdf(pdf_path):
    """Convert PDF using PyMuPDF - much faster and cleaner"""
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]  # Get first page

        # Higher quality rendering
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better quality
        pix = page.get_pixmap(matrix=mat)

        # Save as temporary image
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = str(int(time.time()))
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{base}_{timestamp}_pymupdf.png")
        pix.save(temp_path)

        # Clean up
        pix = None
        doc.close()

        logger.info(f"PDF converted successfully with PyMuPDF: {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"PyMuPDF conversion failed: {e}")
        return None

def convert_pdf_with_pdf2image(pdf_path):
    """Fallback: Convert PDF using pdf2image"""
    try:
        poppler_configs = [
            None,  # System PATH first
            os.path.join(os.path.dirname(__file__), 'poppler-25.07.0', 'Library', 'bin'),
            os.path.join(os.path.dirname(__file__), 'poppler', 'bin'),
        ]
        
        for poppler_path in poppler_configs:
            try:
                kwargs = {
                    'dpi': 200,
                    'first_page': 1,
                    'last_page': 1,
                    'fmt': 'png'
                }
                
                if poppler_path and os.path.exists(poppler_path):
                    kwargs['poppler_path'] = poppler_path
                
                images = convert_from_path(pdf_path, **kwargs)
                
                if images:
                    base = os.path.splitext(os.path.basename(pdf_path))[0]
                    timestamp = str(int(time.time()))
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{base}_{timestamp}_pdf2image.png")
                    images[0].save(temp_path, 'PNG')
                    
                    for img in images:
                        img.close()
                    
                    logger.info(f"PDF converted with pdf2image: {temp_path}")
                    return temp_path
                    
            except Exception as e:
                logger.debug(f"Failed with poppler config {poppler_path}: {e}")
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"pdf2image conversion failed: {e}")
        return None

def convert_pdf_to_image(pdf_path):
    """Smart PDF conversion using best available method"""
    if PDF_METHOD == 'pymupdf':
        return convert_pdf_with_pymupdf(pdf_path)
    elif PDF_METHOD == 'pdf2image':
        return convert_pdf_with_pdf2image(pdf_path)
    else:
        logger.info("No PDF conversion library available")
        return None

def enhance_image(image_path):
    """Enhanced image processing with proper resource management"""
    try:
        base = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = str(int(time.time()))
        enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{base}_{timestamp}_enhanced.jpg")

        with safe_image_context(image_path) as img:
            # Handle different color modes
            if img.mode in ['CMYK', 'LAB']:
                img = img.convert('RGB')
            elif img.mode == 'P':
                img = img.convert('L')
            
            # Smart resizing logic
            width, height = img.size
            max_dimension = max(width, height)
            
            if max_dimension > MAX_DIMENSION:
                scale_factor = MAX_DIMENSION / max_dimension
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized from {width}x{height} to {new_width}x{new_height}")
            elif max_dimension < MIN_DIMENSION:
                scale_factor = MIN_DIMENSION / max_dimension
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Upscaled from {width}x{height} to {new_width}x{new_height}")
            
            # Apply enhancements for CAD drawings
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            # Save enhanced image
            img.save(enhanced_path, 'JPEG', quality=95, optimize=True)
        
        return enhanced_path
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return image_path

def analyze_with_gemini_direct_pdf(pdf_path):
    """Analyze PDF directly with Gemini - optimized for speed"""
    try:
        if not model or not client:
            return {"fields": [{"name": "Error", "value": "Gemini API not configured"}]}

        # Check file size
        file_size = os.path.getsize(pdf_path)
        if file_size > 5 * 1024 * 1024:
            logger.info(f"PDF too large ({file_size/1024/1024:.1f}MB), switching to image conversion")
            return None

        # Read PDF file
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()

        prompt = get_analysis_prompt()

        logger.info("Starting direct PDF analysis...")
        start_time = time.time()

        # Create PDF part using types.Part.from_bytes
        pdf_part = types.Part.from_bytes(
            data=pdf_data,
            mime_type="application/pdf"
        )

        # Try with retry logic for 503 errors
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use structured output with Pydantic model
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt, pdf_part],
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": FieldsModel.model_json_schema(),
                    },
                )

                # --- Dump raw Gemini response for debugging ---
                # try:
                #     raw_dir = os.path.join(os.path.dirname(__file__), 'gemini_raw')
                #     os.makedirs(raw_dir, exist_ok=True)
                #     base = os.path.splitext(os.path.basename(pdf_path))[0]
                #     ts = str(int(time.time()))
                #     raw_path = os.path.join(raw_dir, f"{base}_directpdf_{ts}.json")
                #     with open(raw_path, 'w', encoding='utf-8') as rf:
                #         rf.write(response.text)
                # except Exception as dump_exc:
                #     logger.warning(f"Failed to dump raw Gemini response: {dump_exc}")

                try:
                    fields_result = FieldsModel.model_validate_json(response.text)
                    processing_time = time.time() - start_time
                    logger.info(f"Direct PDF analysis completed in {processing_time:.2f}s (structured, attempt {attempt+1})")
                    return fields_result.model_dump()
                except Exception as e:
                    logger.warning(f"Structured output failed: {e}, falling back to text parsing.")
                    return parse_gemini_response(response.text)

            except Exception as e:
                if "503" in str(e) or "overloaded" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Model overloaded (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                raise  # Re-raise if not 503 or last attempt

        # If all structured attempts fail, try fallback WITHOUT json mode
        logger.warning("Structured output failed, trying text-based fallback...")
        try:
            # Create a simpler prompt for fallback
            fallback_prompt = """Extract ALL information from this technical drawing/document.
            
                                Return your response as a JSON object with this EXACT structure:
                                {
                                "fields": [
                                    {"name": "Field Name 1", "value": "Value 1"},
                                    {"name": "Field Name 2", "value": "Value 2"}
                                ]
                                }

                                Extract EVERY piece of information you see. Be thorough."""

            response = client.models.generate_content(
                model=model,
                contents=[fallback_prompt, pdf_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,  # No JSON mode, let model format naturally
                ),
            )
            processing_time = time.time() - start_time
            logger.info(f"PDF analysis completed in {processing_time:.2f}s (text fallback)")
            return parse_gemini_response(response.text)
            
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return {"fields": [{"name": "Error", "value": f"PDF analysis failed: {str(e)}"}]}
            
    except Exception as e:
        logger.error(f"Direct PDF analysis failed: {e}")
        return {"fields": [{"name": "Error", "value": f"PDF analysis failed: {str(e)}"}]}

def analyze_with_gemini_image(image_path):
    """Analyze image with Gemini Vision - optimized for speed"""
    try:
        if not model or not client:
            return {"fields": [{"name": "Error", "value": "Gemini API not configured"}]}

        logger.info("Starting image analysis...")
        start_time = time.time()

        # Open image with PIL and pass directly
        with safe_image_context(image_path) as img:
            # Resize if needed for faster processing
            width, height = img.size
            if width > 1024 or height > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {img.size} for faster processing")

            prompt = get_analysis_prompt()
            
            # Try with retry logic for 503 errors
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # Use structured output with Pydantic model
                    response = client.models.generate_content(
                        model=model,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            top_p=0.8,
                            top_k=20,
                            max_output_tokens=4096,  # Increased from 2048
                            response_mime_type="application/json",
                            response_schema=FieldsModel,
                        ),
                    )

                    # --- Dump raw Gemini response for debugging ---
                    try:
                        raw_dir = os.path.join(os.path.dirname(__file__), 'gemini_raw')
                        os.makedirs(raw_dir, exist_ok=True)
                        base = os.path.splitext(os.path.basename(image_path))[0]
                        ts = str(int(time.time()))
                        raw_path = os.path.join(raw_dir, f"{base}_image_{ts}.json")
                        with open(raw_path, 'w', encoding='utf-8') as rf:
                            rf.write(response.text)
                    except Exception as dump_exc:
                        logger.warning(f"Failed to dump raw Gemini response: {dump_exc}")

                    try:
                        fields_result = FieldsModel.model_validate_json(response.text)
                        processing_time = time.time() - start_time
                        logger.info(f"Image analysis completed in {processing_time:.2f}s (structured, attempt {attempt+1})")
                        return fields_result.model_dump()
                    except Exception as e:
                        logger.warning(f"Structured output failed: {e}, falling back to text parsing.")
                        return parse_gemini_response(response.text)

                except Exception as e:
                    if "503" in str(e) or "overloaded" in str(e).lower():
                        if attempt < max_retries - 1:
                            logger.warning(f"Model overloaded (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                    raise

            # Fallback without JSON mode
            logger.warning("Structured output failed, trying text-based fallback...")
            fallback_prompt = """Extract ALL information from this technical drawing/document.
            
                                Return your response as a JSON object with this EXACT structure:
                                {
                                "fields": [
                                    {"name": "Field Name 1", "value": "Value 1"},
                                    {"name": "Field Name 2", "value": "Value 2"}
                                ]
                                }

                                Extract EVERY piece of information you see. Be thorough."""

            response = client.models.generate_content(
                model=model,
                contents=[fallback_prompt, img],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            processing_time = time.time() - start_time
            logger.info(f"Image analysis completed in {processing_time:.2f}s (text fallback)")
            return parse_gemini_response(response.text)

    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return {"fields": [{"name": "Error", "value": f"Image analysis failed: {str(e)}"}]}

def get_analysis_prompt():
    """CAD extraction prompt optimized for STRUCTURED OUTPUT."""
    return """
You are an expert CAD and technical drawing analyst.

Your task is to extract ALL visible information from the provided
technical drawing or document with extreme thoroughness.

GENERAL EXTRACTION PRINCIPLES:
- Extract every distinct piece of information you can see
- Each fact must become ONE field with:
  - a clear, descriptive name
    - use names LIKE 'drawing number', 'part number', 'document id', 'doc name', 'version', 'revision', 'title', 'name', 'id', 'number', 'components', 'parts count', 'assembly', 'catalog', 'item', 'model', 'created', 'modified', 'author', 'designer', 'engineer', 'department', 'company', 'date', 'time', 'approved', 'checked', 'drawn by', 'edited', 'updated', 'organization', 'notes', 'comments', 'description', 'text', 'annotation', 'table', 'history', 'remarks', 'specification', 'material', 'finish', 'treatment', 'composition', 'chemical', 'properties','dimension', 'measurement', 'length', 'width', 'height', 'diameter', 'radius', 'angle', 'tolerance', 'units', 'mm', 'inches', 'scale', 'coordinates', 'position', 'location', ONLY IF RELEVANT.
  - the exact value as shown in the document
- Preserve units, symbols, tolerances, and formatting exactly as visible
- If information is unclear or partially visible, prefix the value with "Partial: "

DO NOT summarize.
DO NOT infer missing values.
DO NOT merge unrelated facts into one field.

CATEGORIES OF INFORMATION TO EXTRACT:

DOCUMENT IDENTIFICATION:
- Drawing numbers, part numbers, item numbers
- Revision letters/numbers and revision notes
- Sheet numbers, total sheets, page references
- Drawing titles, part names, assembly names
- Project codes, job numbers, work order numbers
- Drawing scale(s)

METADATA & AUTHORSHIP:
- Designer, drafter, checker, approver names
- Company, department, division names
- Creation dates, revision dates, approval dates
- Drawing standards (ISO, ANSI, DIN, ASME, etc.)
- Related or referenced drawing numbers

DIMENSIONS & MEASUREMENTS:
- All dimensions with units
- Lengths, widths, heights
- Diameters, radii, angles
- Coordinate dimensions
- Reference dimensions
- Tolerances (± values, limit dimensions)
- GD&T symbols and descriptions
- Scale-dependent dimensions

MATERIALS & SPECIFICATIONS:
- Material names and grades
- Material standards (ASTM, SAE, EN, etc.)
- Surface finishes and roughness values
- Coatings, plating, treatments
- Heat treatment requirements
- Weight, mass, density if stated

MANUFACTURING & PROCESS NOTES:
- Machining instructions
- Welding symbols and specifications
- Assembly instructions
- Installation notes
- Inspection requirements
- Quality or process standards

TEXT & ANNOTATIONS:
- All notes, comments, callouts
- Table contents
- BOM-related text (do not restructure, just extract text)
- Warnings, cautions, safety notes
- Legal or compliance notes

TECHNICAL & PERFORMANCE DATA:
- Quantities and counts
- Electrical ratings (voltage, current, power)
- Pressure, temperature, flow ratings
- Serial numbers, batch numbers, lot numbers

GEOMETRIC & VIEW INFORMATION:
- View labels (front, top, side, section, detail)
- Section identifiers
- Datum references
- Centerlines, construction references
- Symbol meanings if explicitly stated

COST & PROCUREMENT (if present):
- Vendor names or codes
- Purchase references
- Cost or pricing text
- Lead times or delivery notes
- Inventory or stock numbers

 EXAMPLE JSON with ALL extracted data:
    {
        "fields": [
            {"name": "Drawing Number", "value": "DWG-12345-Rev-C"},
            {"name": "Part Name", "value": "Motor Housing Assembly"},
            {"name": "Material", "value": "6061-T6 Aluminum Alloy"},
            {"name": "Overall Length", "value": "125.5 ± 0.1 mm"},
            {"name": "Designer", "value": "John Smith"},
            {"name": "Date Created", "value": "2024-01-15"},
            {"name": "Scale", "value": "1:2"},
            {"name": "Surface Finish", "value": "Ra 1.6 μm"},
            {"name": "Tolerance Class", "value": "IT7"},
            {"name": "Manufacturing Process", "value": "CNC Machining"},
            {"name": "Quality Standard", "value": "ISO 9001"},
            {"name": "Weight", "value": "2.3 kg"},
            {"name": "Note 1", "value": "All dimensions in millimeters unless noted"},
            {"name": "Vendor Code", "value": "SUPP-001-ALU"}
        ]
    }

    CRITICAL RULES:
    - Extract EVERY piece of visible text and data
    - Include units with all measurements
    - Capture partial/unclear data with "Partial: " prefix
    - Use descriptive field names that explain what the data represents
    - Include table data, lists, and structured information
    - Don't skip small text, symbols, or annotations
    - Extract data from title blocks, revision blocks, notes sections
    - Include geometric dimensioning and tolerancing (GD&T) symbols
    - Capture all manufacturing and quality specifications
    - Be extremely thorough - aim for 20-50+ fields per drawing
"""

def parse_gemini_response(response_text):
    """Fallback parser for Gemini output - handles various JSON formats."""
    try:
        response_text = response_text.strip()
        logger.info(f"Parsing response of length: {len(response_text)}")
        
        # Remove markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        
        # Try to extract JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            # Clean up trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            result = json.loads(json_str)
            
            # Handle case where result already has "fields" key
            if isinstance(result, dict) and 'fields' in result:
                logger.info(f"Parsed {len(result['fields'])} fields from response")
                return result
            
            # Handle case where result is a flat dict - convert to fields format
            elif isinstance(result, dict):
                logger.info(f"Converting flat dict with {len(result)} keys to fields format")
                fields = []
                for key, value in result.items():
                    if isinstance(value, (str, int, float, bool)):
                        fields.append({"name": key.replace("_", " ").title(), "value": str(value)})
                    elif isinstance(value, dict):
                        # Nested dict - flatten it
                        for sub_key, sub_value in value.items():
                            fields.append({
                                "name": f"{key.replace('_', ' ').title()} - {sub_key.replace('_', ' ').title()}",
                                "value": str(sub_value)
                            })
                    elif isinstance(value, list):
                        # List - join items
                        fields.append({
                            "name": key.replace("_", " ").title(),
                            "value": ", ".join(str(item) for item in value)
                        })
                
                if fields:
                    return {"fields": fields}
        
        # If JSON parsing fails, try to extract key-value pairs from text
        logger.warning("JSON parsing failed, attempting text extraction")
        fields = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Look for key: value patterns
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip(' -•*')
                    value = parts[1].strip(' ,"')
                    if key and value and len(key) < 100 and len(value) < 500:
                        fields.append({"name": key, "value": value})
        
        if fields:
            logger.info(f"Extracted {len(fields)} fields from text parsing")
            return {"fields": fields}
        
        # Last resort: return full text as single field (NO TRUNCATION)
        logger.warning("Could not parse response, returning as single field")
        return {
            "fields": [
                {"name": "Extracted Information", "value": response_text}
            ]
        }
        
    except Exception as e:
        logger.error(f"Parser failed completely: {e}")
        return {
            "fields": [
                {"name": "Raw Analysis", "value": str(response_text)[:2000] + ("..." if len(str(response_text)) > 2000 else "")}
            ]
        }

def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if not file_path:
                continue
            if os.path.exists(file_path):
                name = os.path.basename(file_path).lower()
                if 'temp_' in name:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.debug(f"Could not clean up {file_path}: {e}")

    # Clean up older temp files
    try:
        for fname in os.listdir(app.config['UPLOAD_FOLDER']):
            if fname.startswith('temp_'):
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                try:
                    os.remove(fpath)
                    logger.debug(f"Cleaned up residual: {fpath}")
                except Exception:
                    pass
    except Exception:
        pass

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    temp_files = []
    start_time = time.time()

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        safe_filename = f"{name}_{timestamp}{ext}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)

        analysis_result = None

        if ext.lower() == '.pdf':
            # Try direct PDF analysis first
            logger.info('Attempting direct PDF analysis...')
            analysis_result = analyze_with_gemini_direct_pdf(filepath)

            # Fallback to image conversion
            if not analysis_result or 'Error' in str(analysis_result):
                logger.info('Direct PDF failed, trying image conversion...')
                converted_image = convert_pdf_to_image(filepath)

                if converted_image:
                    temp_files.append(converted_image)
                    enhanced_image = enhance_image(converted_image)
                    temp_files.append(enhanced_image)
                    analysis_result = analyze_with_gemini_image(enhanced_image)
                else:
                    analysis_result = {
                        'fields': [{'name': 'Error', 'value': 'Failed to process PDF'}]
                    }
        else:
            # Process image files
            logger.info('Processing image file...')
            enhanced_image = enhance_image(filepath)
            temp_files.append(enhanced_image)
            analysis_result = analyze_with_gemini_image(enhanced_image)

        # Processing metadata
        processing_time = round(time.time() - start_time, 3)

        metadata = {
            'file_name': safe_filename,
            'file_size': os.path.getsize(filepath),
            'processing_time_seconds': processing_time,
            'confidence': None,
            'language': 'English'
        }

        # Calculate confidence
        try:
            fields_count = len(analysis_result.get('fields', []))
            if fields_count > 0:
                metadata['confidence'] = min(99.9, 80 + fields_count * 1.0)
            else:
                metadata['confidence'] = 50.0
        except Exception:
            metadata['confidence'] = None

        # Save analysis artifact
        analysis_artifact = {
            'fields': analysis_result.get('fields', []),
            'raw': analysis_result,
            'metadata': metadata
        }

        analysis_filename = f"{safe_filename}.analysis.json"
        analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)
        
        with open(analysis_path, 'w', encoding='utf-8') as af:
            json.dump(analysis_artifact, af, indent=2)

        # Clean up temp files
        cleanup_temp_files(*temp_files)

        return jsonify({
            'success': True,
            'message': 'File analyzed successfully!',
            'filename': safe_filename,
            'analysis_file': analysis_filename
        }), 200

    except Exception as e:
        cleanup_temp_files(*temp_files)
        logger.error(f"Upload processing failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analysis/<path:filename>')
def analysis_page(filename):
    """Render analysis page"""
    safe = secure_filename(filename)
    analysis_filename = f"{safe}.analysis.json"
    analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)

    if not os.path.exists(analysis_path):
        flash('Analysis not found', 'error')
        return redirect(url_for('home'))

    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_artifact = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read analysis: {e}")
        flash('Could not read analysis', 'error')
        return redirect(url_for('home'))
    
    if "bom_items" not in analysis_artifact:
        analysis_artifact["bom_items"] = []

    # Build document history
    uploads = []
    try:
        files_with_times = []
        for fname in os.listdir(app.config['UPLOAD_FOLDER']):
            if fname.endswith('.analysis.json'):
                continue
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            mtime = os.path.getmtime(file_path)
            files_with_times.append((fname, mtime))
        
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        
        for fname, _ in files_with_times:
            uploads.append({
                'name': fname,
                'url': url_for('uploaded_file', filename=fname)
            })
    except Exception:
        uploads = []

    metadata = analysis_artifact.get('metadata', {})

    # Get extra file info
    extra_file_info = {}
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if os.path.exists(file_path):
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.pdf' and 'fitz' in globals():
                try:
                    doc = fitz.open(file_path)
                    extra_file_info['pages'] = doc.page_count
                    doc.close()
                except Exception:
                    pass
            elif ext.lower() in ('.png', '.jpg', '.jpeg'):
                try:
                    with Image.open(file_path) as im:
                        extra_file_info['resolution'] = f"{im.width}x{im.height} px"
                except Exception:
                    pass
    except Exception:
        pass

    return render_template('analysis.html', 
                          analysis=analysis_artifact, 
                          uploads=uploads, 
                          metadata=metadata, 
                          extra=extra_file_info, 
                          filename=safe)

@app.route('/save-analysis/<path:filename>', methods=['POST'])
def save_analysis(filename):
    """Save updated analysis"""
    try:
        data = request.get_json()
        if not data or 'fields' not in data:
            return jsonify({'error': 'No fields data'}), 400
        
        safe = secure_filename(filename)
        analysis_filename = f"{safe}.analysis.json"
        analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)
        
        if not os.path.exists(analysis_path):
            return jsonify({'error': 'Analysis file not found'}), 404
        
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_artifact = json.load(f)
        
        analysis_artifact['fields'] = data['fields']
        analysis_artifact['metadata']['last_updated'] = time.time()
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_artifact, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis updated for {filename}")
        
        return jsonify({
            'success': True,
            'message': 'Analysis saved successfully',
            'fields_count': len(data['fields'])
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to save analysis: {e}")
        return jsonify({'error': f'Failed to save: {str(e)}'}), 500

@app.route('/chat/<path:filename>', methods=['POST'])
def chat_endpoint(filename):
    """AI chat endpoint"""
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    safe = secure_filename(filename)
    analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{safe}.analysis.json")
    document_context = load_document_context(analysis_path, safe)
    
    if model and client:
        try:
            prompt = create_enhanced_chat_prompt(document_context, user_message, safe)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return jsonify({'reply': response.text}), 200
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return jsonify({'reply': get_intelligent_fallback_response(user_message, document_context)}), 200

    return jsonify({'reply': get_intelligent_fallback_response(user_message, document_context)}), 200

@app.route('/extract-bom/<path:filename>', methods=['POST'])
def extract_bom(filename):
    """Extract BOM from file"""
    safe = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    bom_result = extract_bom_with_tables(file_path)

    # Save to analysis
    analysis_filename = f"{safe}.analysis.json"
    analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)

    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_artifact = json.load(f)
    else:
        analysis_artifact = {}

    analysis_artifact.setdefault("fields", [])
    analysis_artifact.setdefault("metadata", {})
    analysis_artifact["bom_items"] = bom_result.get("bom_items", [])
    analysis_artifact["bom_raw"] = bom_result.get("bom_raw", "")

    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_artifact, f, indent=2)

    return jsonify({
        'success': True,
        'bom_items': analysis_artifact["bom_items"],
        'total_items': len(analysis_artifact["bom_items"]),
        'raw_preview': analysis_artifact.get("bom_raw", "")[:500]
    }), 200

def load_document_context(analysis_path, filename):
    """Load document context for chat"""
    context = {
        'filename': filename,
        'fields': [],
        'categories': {'A': [], 'B': [], 'C': [], 'D': []},
        'metadata': {},
        'field_count': 0,
        'document_type': 'Unknown'
    }
    
    try:
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                
            fields = analysis_data.get('fields', [])
            context['fields'] = fields
            context['field_count'] = len(fields)
            context['metadata'] = analysis_data.get('metadata', {})
            
            for field in fields:
                category = field.get('category', 'Uncategorized')
                if category in context['categories']:
                    context['categories'][category].append(field)
            
            context['document_type'] = determine_document_type(fields)
    except Exception as e:
        logger.debug(f"Could not load context: {e}")
    
    return context

def determine_document_type(fields):
    """Determine document type"""
    field_text = ' '.join([f.get('name', '') + ' ' + f.get('value', '') for f in fields]).lower()
    
    if any(kw in field_text for kw in ['assembly', 'exploded', 'parts list', 'bom']):
        return 'Assembly Drawing'
    elif any(kw in field_text for kw in ['detail', 'part', 'component']):
        return 'Detail Drawing'
    elif any(kw in field_text for kw in ['schematic', 'circuit', 'electrical']):
        return 'Schematic'
    else:
        return 'Technical Drawing'

def create_enhanced_chat_prompt(context, user_message, filename):
    """Create chat prompt"""
    all_fields = []
    for field in context['fields']:
        all_fields.append(f"- {field['name']}: {field['value']}")
    
    context_summary = '\n'.join(all_fields) if all_fields else "No fields extracted"
    
    return f"""You are an expert CAD/Technical Drawing AI Assistant.

DOCUMENT: {filename} ({context['document_type']})
EXTRACTED DATA ({context['field_count']} fields):
{context_summary}

USER QUESTION: {user_message}

INSTRUCTIONS:
- Answer based on the EXTRACTED DATA above
- If info IS in data, provide it clearly
- If info is NOT in data, say so
- Be helpful and specific
- Use bullet points for lists
- Keep responses concise

RESPONSE:"""

def get_intelligent_fallback_response(user_message, context):
    """Fallback chat response"""
    message_lower = user_message.lower()
    
    # Search for specific info
    if 'part number' in message_lower:
        part_fields = [f for f in context['fields'] 
                      if any(kw in f.get('name', '').lower() 
                            for kw in ['part', 'number', 'item', 'drawing'])]
        if part_fields:
            results = [f"• {f['name']}: {f['value']}" for f in part_fields[:3]]
            return "**Part Number Information:**\n" + '\n'.join(results)
        return "**Part Numbers:** No part numbers found in the document."
    
    elif 'dimension' in message_lower or 'measurement' in message_lower:
        dimensions = [f for f in context['fields'] 
                     if any(kw in f.get('name', '').lower() 
                           for kw in ['dimension', 'length', 'width', 'height', 'diameter'])]
        if dimensions:
            results = [f"• {f['name']}: {f['value']}" for f in dimensions[:4]]
            return "**Dimensions Found:**\n" + '\n'.join(results)
        return "**Dimensions:** No dimensional information was extracted."
    
    # Show available information
    if context['field_count'] > 0:
        sample_fields = context['fields'][:4]
        results = [f"• {f['name']}: {f['value']}" for f in sample_fields]
        return f"**Available Information ({context['field_count']} fields):**\n" + '\n'.join(results) + "\n\nAsk about specific aspects."
    
    return "**Document Analysis:** Being processed. Please try asking about document information."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/test-api')
def test_api():
    """Test API status"""
    return jsonify({
        'status': 'API is working',
        'model_configured': model is not None,
        'gemini_key_present': bool(gemini_api_key)
    })

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        company = request.form.get('company', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()
        
        if not all([name, email, subject, message]):
            flash('Please fill in all required fields.', 'error')
            return redirect(url_for('contact'))
        
        if not (app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']):
            flash('Email service not configured. Contact us at brpuneet898@gmail.com', 'error')
            return redirect(url_for('contact'))
        
        try:
            msg = Message(
                subject=f'ADPA Contact Form: {subject}',
                sender=app.config['MAIL_USERNAME'],
                recipients=['brpuneet898@gmail.com'],
                reply_to=email
            )
            
            msg.body = f"""
New contact form submission:

Name: {name}
Email: {email}
Company: {company or 'Not specified'}
Subject: {subject}

Message:
{message}

---
Reply to: {email}
"""
            
            mail.send(msg)
            flash('Message sent successfully!', 'success')
            
        except Exception as e:
            logger.error(f"Email failed: {e}")
            flash('Error sending message. Email us at brpuneet898@gmail.com', 'error')
        
        return redirect(url_for('contact'))
    
    return render_template("contact.html")

if __name__ == "__main__":
    logger.info(f"Starting application with PDF method: {PDF_METHOD}")
    logger.info(f"Gemini API configured: {bool(client)}")
    app.run(debug=True)