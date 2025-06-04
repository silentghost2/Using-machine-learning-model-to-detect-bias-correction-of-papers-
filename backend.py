from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.generativeai import types as genai_types
import PyPDF2
import os
from groq import Groq
import io
import json
from dotenv import load_dotenv
import re # For regular expressions, likely needed for parsing

load_dotenv()

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv("")
GROQ_API_KEY = os.getenv("")

if not GOOGLE_API_KEY or not GOOGLE_API_KEY.startswith("AIza"):
    print("--- WARNING: GOOGLE_API_KEY is missing or looks invalid in your .env file! ---")
if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    print("--- WARNING: GROQ_API_KEY is missing or looks invalid in your .env file! ---")

try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    if GROQ_API_KEY:
        client_groq = Groq(api_key=GROQ_API_KEY)
    print("Attempted API client configuration.")
except Exception as e:
    print(f"Error during initial API client configuration: {e}")


def extract_text_from_pdf_bytes(pdf_bytes):
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n---PAGE_BREAK---\n" # Add page break marker
    except Exception as e:
        print(f"Error extracting text from PDF bytes: {e}")
    return text

# --- Parsing Functions (These are CRITICAL and need robust implementation) ---
def parse_gemini_page_output(gemini_text_output):
    """
    Parses Gemini's output to extract structured question IDs, student answers,
    and any original marks identified on the page.
    THIS IS A COMPLEX FUNCTION YOU WILL NEED TO WRITE.
    It might involve regex or expecting a specific format from Gemini.
    
    Expected output format:
    [
        {"question_id_on_page": "Q1a", "student_answer_on_page": "...", "original_mark_on_page": "1.5/2 or N/A"},
        {"question_id_on_page": "Q1b", "student_answer_on_page": "...", "original_mark_on_page": "N/A"},
        ...
    ]
    """
    print(f"\n--- Attempting to parse Gemini Page Output ---")
    print(f"Raw Gemini Output (first 500 chars):\n{gemini_text_output[:500]}...")
    
    parsed_qas = []
    # Example of a VERY simple regex approach (highly likely to need improvement)
    # This assumes Gemini outputs "Question ID: Q1a\nStudent Answer: Some text\nOriginal Mark on Page: 1/2"
    # You'll need to adjust this based on your Gemini prompt and its actual output.
    pattern = re.compile(
        r"Question ID:\s*(?P<id>Q\d+[a-zA-Z]?\)?)\s*\n" #  e.g. Q1a) or Q1a
        r"Student Answer:\s*(?P<answer>.*?)\s*\n"
        r"(?:Original Mark on Page:\s*(?P<mark>.*?)\s*\n?)?", # Optional mark
        re.DOTALL | re.MULTILINE
    )

    for match in pattern.finditer(gemini_text_output):
        data = match.groupdict()
        parsed_qas.append({
            "question_id_on_page": data["id"].strip().replace(")", ""), # Normalize ID (e.g., Q1a from Q1a))
            "student_answer_on_page": data["answer"].strip(),
            "original_mark_on_page": data["mark"].strip() if data["mark"] else "N/A"
        })

    if not parsed_qas and gemini_text_output: # Fallback if regex fails but text exists
        print("Warning: Specific Q&A parsing failed. Treating whole Gemini output as one answer for 'UnknownQ'.")
        parsed_qas.append({
            "question_id_on_page": "UnknownQ_from_image", 
            "student_answer_on_page": gemini_text_output.strip(),
            "original_mark_on_page": "N/A"
        })
        
    print(f"Parsed Q&As from Gemini: {parsed_qas}")
    return parsed_qas


def get_specific_ground_truth_for_question(full_ground_truth_text, question_id_target):
    """
    Extracts the relevant section of the ground truth for a specific question_id.
    THIS IS ALSO A COMPLEX FUNCTION YOU WILL NEED TO WRITE.
    It depends on how your ground truth PDF is structured.
    """
    print(f"Searching for ground truth for: {question_id_target} in full GT.")
    # Example: Simple search (you'll need something more robust)
    # This assumes ground truth has clear markers like "Model Answer Q1a:"
    # Normalize question_id_target for searching (e.g. Q1A -> Q1a)
    search_id_normalized = question_id_target.lower().replace(")", "")

    # Try variations for searching
    patterns_to_try = [
        rf"Model Answer\s*(for)?\s*Question\s*{re.escape(search_id_normalized[:-1])}\s*\)?\s*{re.escape(search_id_normalized[-1])}\)?:?\s*(.*?)(?=(Model Answer\s*(for)?\s*Question\s*Q\d|---END_OF_GT_SECTION---|---PAGE_BREAK---))",
        rf"Answer\s*(for)?\s*{re.escape(search_id_normalized)}\s*:?\s*(.*?)(?=(Answer\s*(for)?\s*Q\d|---END_OF_GT_SECTION---|---PAGE_BREAK---))",
        rf"{re.escape(search_id_normalized)}\)\s*(.*?)(?=(Q\d|---END_OF_GT_SECTION---|---PAGE_BREAK---))"
    ]
    
    for pattern_str in patterns_to_try:
        match = re.search(pattern_str, full_ground_truth_text, re.IGNORECASE | re.DOTALL)
        if match:
            print(f"Found GT for {question_id_target} using pattern: {pattern_str[:30]}...")
            return match.group(match.lastindex -1 ).strip() # Get the content part

    print(f"Warning: Could not find specific ground truth for {question_id_target}. Using a generic portion or full GT.")
    # Fallback: return a large chunk or a message
    return f"Specific ground truth for {question_id_target} not reliably parsed. Using general context:\n" + full_ground_truth_text[:1500]


def extract_content_from_image_page_gemini(image_bytes, mime_type):
    if not GOOGLE_API_KEY: return "Gemini API Key not configured."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Analyze this image from a student's handwritten answer sheet.
        Your task is to meticulously identify and extract all distinct question numbers
        (e.g., "Q1 a)", "Question 2.", "Part B Q3b") and the complete handwritten
        answer provided by the student for EACH identified question number.
        If any marks or scores (e.g., "1/2", "5", "-1") are written by a grader directly
        next to or on an answer, please note those down as 'Original Mark on Page'.

        Present your findings in a clear, structured format. For EACH question found:
        Start with "Question ID: [The identified question number]".
        Then, on a new line, "Student Answer: [The student's full answer text for this question]".
        Then, on a new line, "Original Mark on Page: [The mark if visible, otherwise 'N/A']".
        Separate each distinct question-answer block with "---END_OF_QUESTION_BLOCK---".

        If the image is unclear or no distinct questions can be identified, extract all readable text.
        """
        response = model.generate_content(
            [prompt, genai_types.Part(inline_data=genai_types.Blob(mime_type=mime_type, data=image_bytes))],
            generation_config=genai.types.GenerationConfig(temperature=0.1) # Lower temp for more factual extraction
        )
        if response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
            return "Gemini: Content generation blocked by safety filters."
        return response.text
    except Exception as e:
        print(f"Gemini API error (extract_content_from_image_page_gemini): {e}")
        return f"Gemini API Error: {e}"

def reevaluate_with_deepseek(student_answer_text, ground_truth_for_question, question_id, max_marks, original_mark_frontend):
    if not GROQ_API_KEY: return json.dumps({"question_id": question_id, "error": "Groq API Key not configured."})
    
    prompt = f"""
You are an AI Teaching Assistant evaluating a student's answer for question '{question_id}'.
The maximum marks for this question are {max_marks}.
The original mark given by a human grader (entered in the UI) was: {original_mark_frontend if original_mark_frontend is not None else 'Not Provided'}.
The student's answer might also contain a mark written on the paper by an earlier grader, which was identified as part of the "Student's Answer" text below.

Ground Truth / Model Answer for question '{question_id}':
---
{ground_truth_for_question}
---

Student's Answer (extracted from image for question '{question_id}'):
---
{student_answer_text}
---

Task:
1. Carefully compare the "Student's Answer" to the "Ground Truth / Model Answer".
2. Evaluate the student's answer based on: Completeness, Accuracy, Closeness to Model Answer, and Clarity.
3. Re-evaluate and assign a new mark for question '{question_id}' out of {max_marks} marks. Use the general academic marking principles for the given max_marks.
4. Provide a brief explanation for your re-evaluated mark, and specifically comment if it differs significantly from the original mark ({original_mark_frontend if original_mark_frontend is not None else 'N/A'}) or any mark seen on the paper.

Output ONLY a JSON object with the following structure:
{{
  "question_id": "{question_id}",
  "original_mark_acknowledged": "{original_mark_frontend if original_mark_frontend is not None else 'N/A'}",
  "ai_reevaluated_mark": "YOUR_CALCULATED_MARK_AS_A_NUMBER_OR_ERROR_STRING",
  "ai_reasoning": "Your brief explanation for the re-evaluated mark."
}}
Ensure "ai_reevaluated_mark" is a numerical score if successful, or an error message string if evaluation fails.
"""
    try:
        completion = client_groq.chat.completions.create(
            model="deepseek-coder", # Or another suitable model like llama3-70b-8192
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content # This is already a JSON string
    except Exception as e:
        print(f"DeepSeek API error for Q{question_id}: {e}")
        return json.dumps({
            "question_id": question_id,
            "original_mark_acknowledged": original_mark_frontend if original_mark_frontend is not None else 'N/A',
            "ai_reevaluated_mark": "Error",
            "ai_reasoning": f"DeepSeek API Error: {e}"
        })


@app.route('/analyze-paper', methods=['POST'])
def analyze_paper_endpoint():
    print("\n--- Request received at /analyze-paper ---")
    
    ground_truth_pdf_file = request.files.get('ground_truth_pdf')
    student_answer_images = request.files.getlist('student_answer_images')
    original_marks_json_str = request.form.get('original_marks_json')
    question_allocations_json_str = request.form.get('question_allocations_json')

    if not all([ground_truth_pdf_file, student_answer_images, original_marks_json_str, question_allocations_json_str]):
        return jsonify({"error": "Missing one or more required fields (pdf, images, marks, allocations)"}), 400

    try:
        original_marks_frontend = json.loads(original_marks_json_str)
        question_allocations = json.loads(question_allocations_json_str)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON in form data: {e}"}), 400

    print(f"Received {len(student_answer_images)} student image(s).")

    gt_pdf_bytes = ground_truth_pdf_file.read()
    ground_truth_text_full = extract_text_from_pdf_bytes(gt_pdf_bytes)
    if not ground_truth_text_full.strip(): # Check if text extraction actually yielded something
        return jsonify({"error": "Could not extract any text from ground truth PDF. It might be an image-based PDF or corrupted."}), 500

    response_data = {
        "image_evaluations": [], # Evaluations per image page
        "summary_original_total": sum(float(v) for v in original_marks_frontend.values() if v is not None and str(v).strip() != ''),
        "summary_reevaluated_total": 0,
        "discrepancies": [],
        "flagged_for_review": False,
        "reasoning": "Analysis initiated."
    }

    for image_idx, image_file_storage in enumerate(student_answer_images):
        filename = image_file_storage.filename
        print(f"\nProcessing student image {image_idx + 1}/{len(student_answer_images)}: {filename}")
        image_file_storage.seek(0) # Reset pointer
        image_bytes = image_file_storage.read()
        
        gemini_output_for_this_page = extract_content_from_image_page_gemini(image_bytes, image_file_storage.mimetype)
        
        parsed_qas_from_page = parse_gemini_page_output(gemini_output_for_this_page)
        
        image_eval_entry = {
            "filename": filename,
            "gemini_full_extraction": gemini_output_for_this_page,
            "deepseek_question_evaluations": []
        }

        if not parsed_qas_from_page:
            print(f"  No structured Q&As parsed by Gemini for image {filename}. Skipping DeepSeek for this image.")
            image_eval_entry["deepseek_question_evaluations"].append({
                "question_id": "PageLevelError", 
                "error": "Gemini did not identify distinct question-answer blocks for DeepSeek evaluation."
            })
        else:
            for qa_block in parsed_qas_from_page:
                q_id_from_gemini = qa_block["question_id_on_page"].upper() # e.g. Q1A
                student_answer_for_q = qa_block["student_answer_on_page"]
                # original_mark_on_page = qa_block["original_mark_on_page"] # Can be used for context

                # Find max_marks and original_frontend_mark for this q_id_from_gemini
                # This requires normalizing q_id_from_gemini to match keys in question_allocations and original_marks_frontend
                # E.g., if frontend sends "Q1a", Gemini might find "Q1 a)".
                # This is a simplification; you'll need robust key matching.
                
                # Try to find a matching key in allocations (e.g. Q1A in {"Q1a":2, "Q1b":2})
                max_marks_for_q = None
                matched_alloc_key = None
                for alloc_key in question_allocations.keys():
                    if alloc_key.upper() == q_id_from_gemini:
                        max_marks_for_q = question_allocations[alloc_key]
                        matched_alloc_key = alloc_key
                        break
                
                if max_marks_for_q is None:
                    print(f"  Skipping DeepSeek for '{q_id_from_gemini}' (from {filename}): Max marks not found in allocations.")
                    image_eval_entry["deepseek_question_evaluations"].append({
                        "question_id": q_id_from_gemini, 
                        "error": "Max marks for this question not provided in allocations."
                    })
                    continue

                # Try to find original mark from frontend data
                original_mark_key_frontend_a = f"{matched_alloc_key.lower()}_score" # e.g. q1a_score
                original_mark_key_frontend_b = f"{matched_alloc_key.lower()}_score_secb" # e.g. q1a_score_secb
                original_mark_val_frontend = original_marks_frontend.get(original_mark_key_frontend_a, 
                                                                        original_marks_frontend.get(original_mark_key_frontend_b))

                if original_mark_val_frontend is None:
                    print(f"  Skipping DeepSeek for '{q_id_from_gemini}' (from {filename}): Original UI mark not found.")
                    image_eval_entry["deepseek_question_evaluations"].append({
                        "question_id": q_id_from_gemini,
                        "error": "Original mark for this question not provided from UI for AI comparison."
                    })
                    continue

                ground_truth_for_this_q = get_specific_ground_truth_for_question(ground_truth_text_full, q_id_from_gemini)
                
                print(f"  Sending to DeepSeek for '{q_id_from_gemini}' (Max: {max_marks_for_q}, Original UI: {original_mark_val_frontend})")
                deepseek_eval_str = reevaluate_with_deepseek(
                    student_answer_for_q,
                    ground_truth_for_this_q,
                    q_id_from_gemini,
                    max_marks_for_q,
                    original_mark_val_frontend
                )
                try:
                    deepseek_eval_json = json.loads(deepseek_eval_str) # DeepSeek should return JSON string now
                    image_eval_entry["deepseek_question_evaluations"].append(deepseek_eval_json)

                    if "ai_reevaluated_mark" in deepseek_eval_json and str(deepseek_eval_json["ai_reevaluated_mark"]).lower() != "error":
                        try:
                            ai_mark = float(deepseek_eval_json["ai_reevaluated_mark"])
                            response_data["summary_reevaluated_total"] += ai_mark
                            if original_mark_val_frontend is not None and abs(float(original_mark_val_frontend) - ai_mark) > 0.01:
                                deepseek_eval_json["discrepancy"] = True
                                response_data["discrepancies"].append({
                                    "image_filename": filename,
                                    "question_id": q_id_from_gemini,
                                    "original_mark": original_mark_val_frontend,
                                    "ai_mark": ai_mark,
                                    "reason": deepseek_eval_json.get("ai_reasoning", "N/A")
                                })
                            else:
                                deepseek_eval_json["discrepancy"] = False
                        except ValueError:
                            print(f"  Error: AI reevaluated mark '{deepseek_eval_json['ai_reevaluated_mark']}' for {q_id_from_gemini} is not a valid number.")
                            deepseek_eval_json["error"] = "AI returned a non-numerical mark."
                    else:
                        print(f"  DeepSeek error or no valid mark for {q_id_from_gemini}: {deepseek_eval_json.get('ai_reasoning', 'Unknown DeepSeek issue')}")

                except json.JSONDecodeError as e:
                    print(f"  Error parsing DeepSeek JSON for {q_id_from_gemini}: {e}. Raw output: {deepseek_eval_str}")
                    image_eval_entry["deepseek_question_evaluations"].append({
                        "question_id": q_id_from_gemini,
                        "error": "Failed to parse DeepSeek's JSON response.",
                        "raw_deepseek_output": deepseek_eval_str
                    })
        
        response_data["image_evaluations"].append(image_eval_entry)

    response_data["flagged_for_review"] = len(response_data["discrepancies"]) > 0
    response_data["reasoning"] = f"{len(response_data['discrepancies'])} potential discrepancies found by AI." if response_data["flagged_for_review"] else "AI analysis complete. No major discrepancies found against original marks."

    print(f"--- Sending response to frontend: ---\n{json.dumps(response_data, indent=2)}\n------------------------------------")
    return jsonify(response_data)


if __name__ == '__main__':
    print("===================================================")
    print("  Starting AI Grading Backend Server (Flask)       ")
    print("  Listening on http://127.0.0.1:5001/analyze-paper")
    print("  Make sure your .env file has GOOGLE_API_KEY and GROQ_API_KEY")
    print("  Press CTRL+C to stop the server.                 ")
    print("===================================================")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)