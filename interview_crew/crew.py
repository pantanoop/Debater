import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class DebaterCrew:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")
        
        print(f"🔑 Gemini API Key found: {self.gemini_key[:10]}...")
        genai.configure(api_key=self.gemini_key)

    # ----------------------------------------------------------------------
    # 1️⃣ GENERATE PRO ARGUMENTS
    # ----------------------------------------------------------------------
    def generate_pro_arguments(self, topic: str) -> list:
        print(f"🟩 Generating PRO arguments for: {topic}")

        models_to_try = [
            'gemini-2.0-flash',
            'gemini-2.0-flash-001',
            'gemini-2.0-flash-lite',
            'gemini-pro-latest'
        ]

        for model in models_to_try:
            try:
                print(f"🔄 Trying model: {model}")
                m = genai.GenerativeModel(model)

                prompt = f"""
                You are a PRO debater. Argue IN FAVOR of this topic:

                Topic: "{topic}"

                Generate 6–8 strong arguments.

                Return ONLY JSON:
                [
                    {{"argument": "Point 1"}},
                    {{"argument": "Point 2"}}
                ]
                """

                response = m.generate_content(prompt)

                if response and response.text:
                    text = response.text.strip()

                    # Extract JSON
                    start = text.find("[")
                    end = text.rfind("]") + 1

                    if start != -1 and end != -1:
                        data = json.loads(text[start:end])
                        print("✅ PRO arguments generated")
                        return data

            except Exception as e:
                print(f"❌ Model failed {model}: {str(e)[:80]}")

        print("⚠️ All models failed → Using fallback PRO arguments")
        return self.get_fallback_pro(topic)

    def get_fallback_pro(self, topic: str):
        return [
            {"argument": f"Supporting point 1 for '{topic}'"},
            {"argument": f"Supporting point 2 for '{topic}'"},
            {"argument": f"Supporting point 3 for '{topic}'"},
            {"argument": f"Supporting point 4 for '{topic}'"}
        ]

    # ----------------------------------------------------------------------
    # 2️⃣ GENERATE CON ARGUMENTS
    # ----------------------------------------------------------------------
    def generate_con_arguments(self, topic: str) -> list:
        print(f"🟥 Generating CON arguments for: {topic}")

        models_to_try = [
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-pro-latest'
        ]

        for model in models_to_try:
            try:
                print(f"🔄 Trying model: {model}")
                m = genai.GenerativeModel(model)

                prompt = f"""
                You are a CON debater. Argue AGAINST this topic:

                Topic: "{topic}"

                Generate 6–8 strong opposing arguments.

                Return ONLY JSON:
                [
                    {{"argument": "Point 1"}},
                    {{"argument": "Point 2"}}
                ]
                """

                response = m.generate_content(prompt)

                if response and response.text:
                    text = response.text.strip()

                    start = text.find("[")
                    end = text.rfind("]") + 1

                    if start != -1 and end != -1:
                        data = json.loads(text[start:end])
                        print("✅ CON arguments generated")
                        return data

            except Exception as e:
                print(f"❌ Model failed {model}: {str(e)[:80]}")

        print("⚠️ All models failed → Using fallback CON arguments")
        return self.get_fallback_con(topic)

    def get_fallback_con(self, topic: str):
        return [
            {"argument": f"Opposing point 1 for '{topic}'"},
            {"argument": f"Opposing point 2 for '{topic}'"},
            {"argument": f"Opposing point 3 for '{topic}'"},
            {"argument": f"Opposing point 4 for '{topic}'"}
        ]
    
    # ----------------------------------------------------------------------
    # 2.5️⃣ GENERATE OPPONENT ARGUMENTS BASED ON USER SIDE
    # ----------------------------------------------------------------------
    def generate_ai_opponent_arguments(self, user_side: str, topic: str):
        """
        When user chooses a side (PRO/CON),
        AI will automatically generate arguments for the opposite side.
        """
        user_side = user_side.lower().strip()

        if user_side == "pro":
            print("🟥 User is PRO → AI will argue AGAINST the topic.")
            return self.generate_con_arguments(topic)

        elif user_side == "con":
            print("🟩 User is CON → AI will argue IN FAVOR of the topic.")
            return self.generate_pro_arguments(topic)

        else:
            raise ValueError("Side must be 'pro' or 'con'")

    # ----------------------------------------------------------------------
    # 3️⃣ JUDGE THE DEBATE
    # ----------------------------------------------------------------------
    def judge_debate(self, topic: str, pro: list, con: list) -> dict:
        print("⚖️ Evaluating debate arguments...")

        models_to_try = [
         "models/gemini-2.5-flash",      # working model
         "models/gemini-pro-latest"      # fallback to latest pro model
        ]

        pro_text = "\n".join([f"- {p['argument']}" for p in pro])
        con_text = "\n".join([f"- {c['argument']}" for c in con])

        for model in models_to_try:
            try:
                m = genai.GenerativeModel(model)

                prompt = f"""
                You are the JUDGE of a debate on:

                "{topic}"

                ### PRO ARGUMENTS:
                {pro_text}

                ### CON ARGUMENTS:
                {con_text}

                Evaluate both sides and produce a JSON report:

                {{
                    "winner": "PRO or CON",
                    "reason": "Why this side won",
                    "strengths_pro": ["point1", "point2"],
                    "strengths_con": ["point1", "point2"],
                    "summary": "Short debate summary"
                }}

                Return ONLY JSON.
                """

                response = m.generate_content(prompt)

                if response and response.text:
                    text = response.text.strip()

                    start = text.find("{")
                    end = text.rfind("}") + 1

                    if start != -1 and end != -1:
                        data = json.loads(text[start:end])
                        print("🏆 Judge evaluation complete")
                        return data

            except Exception as e:
                print(f"❌ Judge model failed {model}: {str(e)[:80]}")

        print("⚠️ All judge models failed → Using fallback judge decision")
        return self.get_fallback_judgement(topic, pro, con)

    def get_fallback_judgement(self, topic, pro, con):
        winner = "PRO" if len(pro) >= len(con) else "CON"

        return {
            "winner": winner,
            "reason": "Fallback scoring based on number of arguments provided.",
            "strengths_pro": [p["argument"] for p in pro[:2]],
            "strengths_con": [c["argument"] for c in con[:2]],
            "summary": f"A simplified fallback judgement for debate on '{topic}'."
        }
