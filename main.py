import os, base64, json, secrets
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

app = FastAPI(title='OZC V89 Drawing Analyzer')
origins = [x.strip() for x in os.getenv('ALLOWED_ORIGINS', '').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['POST', 'GET'], allow_headers=['*'], allow_credentials=False)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/analyze')
async def analyze(image: UploadFile = File(...), x_ozc_access_key: str | None = Header(default=None)):
    access_key = os.getenv('OZC_ACCESS_KEY')
    if not access_key:
        raise HTTPException(503, 'Access control not configured')
    if not x_ozc_access_key or not secrets.compare_digest(x_ozc_access_key, access_key):
        raise HTTPException(401, 'Unauthorized')
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise HTTPException(503, 'OPENAI_API_KEY not configured')
    if image.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(415, 'JPEG, PNG or WEBP required')
    content = await image.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, 'Maximum image size 10MB')
    data_url = 'data:' + image.content_type + ';base64,' + base64.b64encode(content).decode()
    client = AsyncOpenAI(api_key=key)
    try:
        response = await client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': ('You read architectural aluminum window/door drawings. Return only JSON with keys customer, project, system, glass, positions, warnings. positions is array of objects with width, height, records, pieces. Dimensions must be centimeters, outside-to-outside. If dimensions are missing or uncertain, do not guess; omit that position and explain in warnings. records is integer count of visible vertical mullions, pieces may be empty array. system is c60 or aldoks only if clear, otherwise c60. Do not infer dimensions from drawing proportions.')},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Extract clearly written outside dimensions and visible vertical records. If any ambiguous labels, add warnings in Turkish. Never invent measurements.'},
                    {'type': 'image_url', 'image_url': {'url': data_url, 'detail': 'high'}}
                ]}
            ], temperature=0, max_tokens=1500
        )
        return {'result': json.loads(response.choices[0].message.content)}
    except Exception:
        raise HTTPException(502, 'Analysis failed; check server logs')
