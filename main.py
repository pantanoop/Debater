# main.py - Updated stable version with robust Gemini handling + PDF export
import os
import io
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, session, redirect, url_for, send_file
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter

# -----------------------------
# Configuration / Setup
# -----------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "harman_secret")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)
print(f"🔑 Gemini loaded: {GEMINI_API_KEY[:8]}...")

# List of models to try (use the names you discovered are available)
WORKING_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-embedding-001"
]

# -----------------------------
# Robust extractor for responses
# -----------------------------
# ...existing code...
def extract_text(resp):
    """
    Robust extractor: try many common SDK response shapes.
    Prints a small debug summary so you can inspect actual structure.
    """
    if not resp:
        return None

    # quick debug summary
    try:
        print("   DEBUG resp repr:", type(resp))
    except Exception:
        pass

    # helper to safely get attributes or dict keys
    def get(obj, *keys):
        cur = obj
        for k in keys:
            if cur is None:
                return None
            # dict-like
            try:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                    continue
            except Exception:
                pass
            # attr-like
            try:
                cur = getattr(cur, k)
                continue
            except Exception:
                pass
            # sequence index
            try:
                if isinstance(cur, (list, tuple)) and isinstance(k, int) and 0 <= k < len(cur):
                    cur = cur[k]
                    continue
            except Exception:
                pass
            return None
        return cur

    # Try many likely places in order
    candidates = get(resp, "candidates") or get(resp, "result", "candidates") or get(resp, "output") or get(resp, "result", "output")
    # If candidates is a dict/wrapper with 'content'
    if not candidates and isinstance(resp, dict):
        # look for keys that often contain text
        for key in ("output_text", "text", "content", "response"):
            if key in resp:
                val = resp[key]
                if isinstance(val, str) and val.strip():
                    return val.strip()

    # If candidates exists, try to extract
    if candidates:
        # iterate candidate objects
        try:
            for c in candidates:
                # candidate.content can be list of parts or a dict
                content = get(c, "content") or get(c, "message") or c
                # content may be a list of parts or a dict/text
                if isinstance(content, (list, tuple)):
                    text_chunks = []
                    for part in content:
                        # part might be dict with 'text' or 'content' with 'text'
                        t = None
                        if isinstance(part, dict):
                            t = part.get("text") or part.get("content") or part.get("output") or part.get("output_text")
                        else:
                            t = getattr(part, "text", None) or getattr(part, "content", None)
                        if isinstance(t, str) and t.strip():
                            text_chunks.append(t)
                    if text_chunks:
                        return "".join(text_chunks).strip()
                elif isinstance(content, dict):
                    # direct dict content
                    for k in ("text", "output_text", "content", "message", "response"):
                        if k in content and isinstance(content[k], str) and content[k].strip():
                            return content[k].strip()
                elif isinstance(content, str) and content.strip():
                    return content.strip()

                # fallback candidate.text
                cand_text = get(c, "text")
                if isinstance(cand_text, str) and cand_text.strip():
                    return cand_text.strip()
        except Exception as e:
            print("   extract_text candidate parsing error:", e)

    # Extra fallbacks - check some top-level fields
    for key in ("output_text", "text", "response", "content"):
        val = None
        try:
            val = getattr(resp, key) if hasattr(resp, key) else (resp.get(key) if isinstance(resp, dict) else None)
        except Exception:
            val = None
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Maybe it's nested under result.output_text
    try:
        v = getattr(getattr(resp, "result", None), "output_text", None) or (resp.get("result", {}).get("output_text") if isinstance(resp, dict) else None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass

    return None



# ...existing code...
def model_generate(prompt: str, max_tokens=400, temperature=0.7):
    """
    Try models in WORKING_MODELS order. If generation is cut off by MAX_TOKENS
    we do one bounded retry with increased max_tokens. If a quota error occurs,
    stop and return None so the app can show fallback text.
    """
    print("🚀 model_generate CALLED")
    for model_name in WORKING_MODELS:
        tried_increase = False
        current_max = max_tokens
        for attempt in (1, 2):  # allow one retry with larger token budget
            try:
                print(f"⚡ Trying model: {model_name} (max_tokens={current_max})")

                model = genai.GenerativeModel(
                    model_name,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": current_max
                    }
                )

                response = model.generate_content(prompt)

                # Debug-print the response type and a short serialized sample
                try:
                    if hasattr(response, "to_dict"):
                        sample = response.to_dict()
                    else:
                        sample = json.loads(json.dumps(response, default=lambda o: getattr(o, "__dict__", str(o))))
                    print("   DEBUG response keys sample:", list(sample.keys())[:10])
                except Exception as e:
                    print("   DEBUG could not serialize response:", e)

                text = extract_text(response)
                if text:
                    print("✅ AI reply generated successfully (model:", model_name, ")")
                    return text

                # Inspect finish reasons
                try:
                    candidates = getattr(response, "candidates", None) or getattr(getattr(response, "result", None) or {}, "candidates", None)
                    if candidates:
                        frs = [getattr(c, "finish_reason", None) for c in candidates]
                        print(f"   No parts found — finish reasons: {frs}")
                        # If truncated by MAX_TOKENS, allow one retry with larger max_tokens
                        if all(str(f).endswith("MAX_TOKENS") for f in frs) and not tried_increase:
                            tried_increase = True
                            current_max = min(current_max * 2, 1200)  # cap to limit token consumption
                            print(f"   Detected MAX_TOKENS; retrying once with max_tokens={current_max}")
                            continue
                    else:
                        print("   No candidates in response.")
                except Exception as e:
                    print("   Could not read finish reasons:", e)

                print(f"❌ No text returned by {model_name}; trying next model.")
                break  # break out of retry loop and try next model

            except Exception as e:
                msg = repr(e)
                # Quota / resource exhausted: stop trying other models and surface fallback
                if "Quota exceeded" in msg or "ResourceExhausted" in msg or "quota" in msg.lower():
                    print(f"❌ MODEL ERROR (quota) for {model_name}: {msg}")
                    print("❌ Quota problem detected. Enable billing or remove preview models. Stopping attempts.")
                    return None
                print(f"❌ MODEL ERROR ({model_name}): {msg}")
                # On other errors, try next configured model
                break

    print("❌ ALL MODELS FAILED")
    return None
# ...existing code...

# -----------------------------
# Helper: truncate / summarize context
# -----------------------------
def build_context(transcript, keep_last_n=2):
    """
    Build a short context string from the transcript.
    keep_last_n: number of recent entries to include (default 2).
    """
    recent = transcript[-keep_last_n:]
    context_lines = [f"{t['speaker']}: {t['text']}" for t in recent]
    return "\n".join(context_lines)

# -----------------------------
# Core AI functions
# -----------------------------
def generate_ai_reply(topic, ai_side, transcript, last_user_text, round_num):
    """
    Generate AI reply with robust handling.
    If all models fail or return nothing, fallback text is used.
    Keeps context short to avoid hitting token limits.
    """
    # Keep only the last 1 or 2 entries to reduce prompt size
    context = build_context(transcript, keep_last_n=1)

    prompt = f"""
You are an AI debater assigned the side: {ai_side}.
Topic: "{topic}"

Context:
{context}

Human just argued:
\"{last_user_text}\"

Reply concisely as the {ai_side} side. Keep the reply between 100 and 220 words.
"""

    # Try generating reply
    reply = model_generate(prompt, max_tokens=400, temperature=0.7)

    # Fallback if AI fails
    if not reply:
        reply = "(AI could not generate a response for this round.)"

    return reply

# -----------------------------
# AI Judge Function (FIXED)
# -----------------------------
def judge_transcript(topic, transcript):
    """
    Uses Gemini to judge the full debate and returns a structured dict.
    Always returns safe defaults even if AI fails.
    """

    # Build transcript text safely
    convo = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])

    prompt = f"""
You are a professional debate judge.

Topic: {topic}

Debate Transcript:
{convo}

Return ONLY valid JSON in this exact format:

{{
  "winner": "PRO or CON or TIE",
  "reason": "Short explanation",
  "scores": {{
    "PRO": 0-10,
    "CON": 0-10
  }},
  "strengths": {{
    "PRO": ["point1", "point2", "point3"],
    "CON": ["point1", "point2", "point3"]
  }},
  "summary": "Final overall judgement"
}}
"""

    try:
        response = model_generate(prompt, max_tokens=600, temperature=0.4)

        if not response:
            raise ValueError("Empty AI response")

        # Try to extract JSON safely
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        data = json.loads(response[json_start:json_end])

        return data

    except Exception as e:
        print("❌ Judge AI failed:", e)

        # ✅ SAFE FALLBACK (prevents 500 crash)
        return {
            "winner": "TIE",
            "reason": "AI judge could not evaluate the debate.",
            "scores": {"PRO": 5.0, "CON": 5.0},
            "strengths": {
                "PRO": ["N/A", "N/A", "N/A"],
                "CON": ["N/A", "N/A", "N/A"]
            },
            "summary": "Automatic fallback judgement used due to AI failure."
        }

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    # clear session state for a fresh start
    session.clear()
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    topic = request.form.get("topic", "").strip()
    user_side = request.form.get("side", "").strip().upper()

    if not topic or user_side not in ("PRO", "CON"):
        return redirect(url_for("index"))

    ai_side = "CON" if user_side == "PRO" else "PRO"

    session["topic"] = topic
    session["user_side"] = user_side
    session["ai_side"] = ai_side
    session["round"] = 0
    session["transcript"] = []
    # initialize debate history (one object per round) with timestamps
    session["debate_history"] = []
    session.modified = True

    return redirect(url_for("debate"))

@app.route("/debate", methods=["GET", "POST"])
def debate():
    if "topic" not in session:
        return redirect(url_for("index"))

    topic = session["topic"]
    user_side = session["user_side"]
    ai_side = session["ai_side"]
    transcript = session.get("transcript", [])
    current_round = session.get("round", 0)
    debate_history = session.get("debate_history", [])

    if request.method == "POST":
        user_text = request.form.get("argument", "").strip()
        if not user_text:
            return redirect(url_for("debate"))

        # Append user's message
        transcript.append({
            "speaker": user_side,
            "text": user_text,
            "round": current_round + 1
        })

        # Generate AI reply safely
        ai_reply = generate_ai_reply(topic, ai_side, transcript, user_text, current_round + 1)

        # Append AI reply
        transcript.append({
            "speaker": ai_side,
            "text": ai_reply,
            "round": current_round + 1
        })

        # Update session
        session["transcript"] = transcript
        session["round"] = current_round + 1

        # Append to debate_history with timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_round = {
            "round": current_round + 1,
            "user_role": user_side,
            "user_text": user_text,
            "ai_text": ai_reply,
            "timestamp": ts
        }
        debate_history.append(new_round)
        session["debate_history"] = debate_history
        session.modified = True

        return redirect(url_for("debate"))

    # Ensure debate_history exists for GET
    if "debate_history" not in session:
        session["debate_history"] = []
        session.modified = True

    return render_template(
        "debate.html",
        topic=topic,
        transcript=transcript,
        round=current_round,
        user_side=user_side,
        ai_side=ai_side,
        debate_history=session.get("debate_history", [])
    )

@app.route("/finish", methods=["GET", "POST"])
def finish():
    if "topic" not in session or not session.get("transcript"):
        return redirect(url_for("index"))

    topic = session["topic"]
    transcript = session.get("transcript", [])

    # Get AI judge report
    report = judge_transcript(topic, transcript)

    # Safely extract data with defaults
    winner = report.get("winner", "TIE")
    reason = report.get("reason", "No reason provided.")
    summary = report.get("summary", "No summary available.")

    scores = report.get("scores", {})
    PRO_score = scores.get("PRO", 0.0)
    CON_score = scores.get("CON", 0.0)

    strengths = report.get("strengths", {})
    strengths_PRO = strengths.get("PRO", [])
    strengths_CON = strengths.get("CON", [])

    # Ensure at least 3 entries to avoid IndexError
    strengths_PRO += ["N/A"] * (3 - len(strengths_PRO))
    strengths_CON += ["N/A"] * (3 - len(strengths_CON))

    # Build report text
    report_text = f"""
AI DEBATE JUDGEMENT REPORT
---------------------------------------

Topic: {topic}

---------------------------------------
FINAL VERDICT
---------------------------------------

Winner: {winner}

Reason:
{reason}

---------------------------------------
SCORES
---------------------------------------
PRO: {PRO_score}
CON: {CON_score}

---------------------------------------
STRENGTHS OF PRO
---------------------------------------
- {strengths_PRO[0]}
- {strengths_PRO[1]}
- {strengths_PRO[2]}

---------------------------------------
STRENGTHS OF CON
---------------------------------------
- {strengths_CON[0]}
- {strengths_CON[1]}
- {strengths_CON[2]}

---------------------------------------
OVERALL SUMMARY
---------------------------------------
{summary}

---------------------------------------
Generated by AI Debate System
"""

    return render_template("final_report.html", report=report_text)


@app.route("/restart")
def restart():
    session.clear()
    return redirect(url_for("index"))

# -----------------------------
# PDF export route (/download_pdf)
# -----------------------------
@app.route("/download_pdf")
def download_pdf():
    if "debate_history" not in session or not session.get("debate_history"):
        return redirect(url_for("debate"))

    topic = session.get("topic", "No topic")
    user_side = session.get("user_side", "USER")
    ai_side = session.get("ai_side", "AI")
    history = session.get("debate_history", [])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    small = ParagraphStyle("small", parent=body_style, fontSize=9)

    # Header
    story.append(Paragraph("AI Debate Transcript", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Topic:</b> {topic}", heading_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>User side:</b> {user_side}    <b>AI side:</b> {ai_side}", body_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Spacer(1, 12))

    # Content: each round
    for item in history:
        story.append(Paragraph(f"<b>Round {item['round']} — {item.get('timestamp','')}</b>", styles["Heading3"]))
        story.append(Spacer(1, 4))

        story.append(Paragraph(f"<b>User ({item['user_role']}):</b>", body_style))
        story.append(Paragraph(item['user_text'].replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>AI:</b>", body_style))
        story.append(Paragraph(item['ai_text'].replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    buffer.seek(0)

    filename = f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
