from openai import OpenAI
import json
import os
from datetime import datetime
from .prompts import SYSTEM_PROMPT, build_user_message
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from .config import DEFAULT_MODEL, DECISIONS_DIR, PROMPTS_DIR
from .logger import logger

# Load environment variables from .env file
# Assumes .env is in the parent directory of src
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Note: Replace 'your-api-key' with the actual API key or use environment variable
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")
model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)

# Initialize client with optional base_url
client_kwargs = {"api_key": api_key}
if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

# Define Response Schema
class DecisionSchema(BaseModel):
    decision: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    target_price: float
    stop_loss: float
    take_profit: float

def get_ai_decision(etf_code, df, prompt_file_template='prompts/{date}_{etf}_prompt.txt',
                   decision_file_template='decisions/{date}_{etf}.json'):
    """获取AI决策"""
    system_prompt = SYSTEM_PROMPT
    user_message = build_user_message(etf_code, df)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Parse and Validate
        try:
            raw_decision = json.loads(content)
            validated_decision = DecisionSchema(**raw_decision)
            decision = validated_decision.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Decision validation error: {e}")
            # Fallback or Retry Logic could go here
            # For now, return a safe hold if validation fails significantly
            return {
                "decision": "hold",
                "confidence": 0.0,
                "reasoning": f"AI返回格式错误: {str(e)}",
                "target_price": 0,
                "stop_loss": 0,
                "take_profit": 0
            }

        date_str = datetime.now().strftime('%Y%m%d')
        prompt_path = os.path.join(PROMPTS_DIR, f"{date_str}_{etf_code}_prompt.txt")
        decision_path = os.path.join(DECISIONS_DIR, f"{date_str}_{etf_code}.json")

        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(f"System Prompt:\n{system_prompt}\n\nUser Message:\n{user_message}")
            
        with open(decision_path, 'w', encoding='utf-8') as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
            
        return decision
        
    except Exception as e:
        logger.error(f"Error getting AI decision: {e}")
        # Return a safe default decision
        return {
            "decision": "hold",
            "confidence": 0.0,
            "reasoning": f"AI调用失败: {str(e)}",
            "target_price": 0,
            "stop_loss": 0,
            "take_profit": 0
        }
