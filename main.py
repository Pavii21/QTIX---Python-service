from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import spacy
from spacy.matcher import Matcher
import re
import uvicorn

class EvaluationRequest(BaseModel):
    response: str
    required_keywords: List[str]

class EvaluationResult(BaseModel):
    scores: dict
    feedback: List[str]
    metrics: dict

class HRResponseEvaluator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        
        filler_patterns = [
            [{"LOWER": "um"}], [{"LOWER": "uh"}], [{"LOWER": "like"}],
            [{"LOWER": "you"}, {"LOWER": "know"}],
            [{"LOWER": "basically"}], [{"LOWER": "literally"}]
        ]
        self.matcher.add("FILLER_WORDS", filler_patterns)

        self.w_a = 0.4  # Relevance
        self.w_b = 0.3  # Clarity
        self.w_c = 0.3  # Depth

    def preprocess(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def evaluate(self, response_text: str, keywords: List[str]) -> dict:
        doc = self.nlp(self.preprocess(response_text))
        
        lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        matches = sum(1 for kw in keywords if kw.lower() in lemmas)
        relevance_score = min((matches / len(keywords)) * 1.5, 1.0) if keywords else 1.0
        
        filler_count = len(self.matcher(doc))
        sentences = list(doc.sents)
        avg_len = len(doc) / len(sentences) if sentences else 0
        penalty = (filler_count * 0.05) + (0.2 if avg_len > 30 else 0)
        clarity_score = max(1.0 - penalty, 0.0)
        
        star_keywords = {"situation", "task", "action", "result", "team", "goal", "achieved", "improved", "managed"}
        star_matches = len(star_keywords.intersection(set(lemmas)))
        depth_score = (min(len(doc) / 80.0, 1.0) * 0.6) + (min(star_matches / 3.0, 1.0) * 0.4)
        
        q_hr = (self.w_a * relevance_score) + (self.w_b * clarity_score) + (self.w_c * depth_score)
        
        feedback = []
        if relevance_score < 0.6: feedback.append("Try to directly address the core concepts of the question.")
        if clarity_score < 0.7: feedback.append(f"We noticed {filler_count} filler words. Try using more concise phrasing.")
        if depth_score < 0.6: feedback.append("Provide more context. Use the STAR method to structure your examples.")
        if not feedback: feedback.append("Excellent response! Clear, relevant, and well-structured.")

        return {
            "scores": {
                "relevance": round(relevance_score, 2),
                "clarity": round(clarity_score, 2),
                "depth": round(depth_score, 2),
                "overall": round(q_hr, 3)
            },
            "feedback": feedback,
            "metrics": {"word_count": len(doc), "fillers": filler_count}
        }

app = FastAPI(title="HR NLP Evaluation Service")
evaluator = HRResponseEvaluator()

@app.post("/api/evaluate", response_model=EvaluationResult)
async def evaluate_endpoint(request: EvaluationRequest):
    try:
        result = evaluator.evaluate(request.response, request.required_keywords)
        return EvaluationResult(
            scores=result["scores"],
            feedback=result["feedback"],
            metrics=result["metrics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
