import os, base64, json, secrets, math, logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

VERSION = 'ozc-v122-openings'
logger = logging.getLogger('uvicorn.error')
app = FastAPI(title='OZC Drawing Analyzer', version=VERSION)
origins = [x.strip() for x in os.getenv('ALLOWED_ORIGINS', '').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['POST', 'GET'], allow_headers=['*'])

INSTRUCTIONS = '''You read Turkish hand-drawn aluminium joinery sheets. Return JSON only.
Read ALL positions in visual order, left to right and top to bottom. Ignore material stock totals as positions.
Extract THREE separate things for EVERY position: (1) written outside width/height in cm, (2) frame/mullion layout, (3) opening symbols and system labels. Do not stop after measuring rectangles.
User drawing convention: one chevron/triangle identifies one operable leaf (door or sash). Opposing chevrons or an X inside an operable opening denote BINILI double leaves: doubleSash for binili kanat, double for binili kapi. This is not a fixed-glass mullion. Show two leaves with no fixed middle mullion. A rectangle above leaves is fixed transom glass; do not turn the entire opening into a full-height sash. A neighboring rectangle is fixed glazing. Crossed-out text/dimensions are corrections, not opening symbols. 'Sonra imalat' is a production note, not a reason to discard an opening.
Respect labels local to each position: Aldoks written above a WC door applies to that door only. C60/RW60 and Aldoks must never be merged into one system for all positions. Labels may apply to neighboring similar doors. If Aldoks series (tam/yarim) is absent, report system=aldoks and system_series=null, plus a Turkish warning. Do not guess profile series.
Top-level keys: customer, project, system (c60/aldoks/null), glass, positions, warnings (Turkish strings).
Each position: width, height, system (c60/aldoks/null), system_series (tam/yarim/null), records, pieces, warnings, note.
width/height must come from clearly written outside dimensions, never proportions. Use null for unreadable dimensions and warn; retain original visual order.
records counts fixed vertical mullions only, NOT sash stiles, double-leaf meeting stiles, frames or dimension ticks.
pieces contains all visible fixed mullions AND all openings. Allowed kind: v,h,hv,hh,door,double,sash,doubleSash. Normalized x/y are visual layout coordinates (0..1) only. Opening pieces require bounds={left,right,top,bottom} describing the visible opening within the outer frame (0..1), align=left/center/right, side=left/right, mainSide=right unless explicitly indicated. doorWidth/doorHeight must be explicit written internal opening dimensions, otherwise null. Flag schematic=true when bounds are estimated from sketch proportions. These proportions are for schematic display only, never manufacturing measurements. Never invent 90/180/220 dimensions. v/h pieces use x/y; hh also includes doorSpan=[left,right]. Do not duplicate an opening edge if the shared fixed mullion is already represented. If an opening symbol exists, pieces MUST NOT be empty. If type is uncertain, preserve your best structural classification and explain in Turkish warnings, rather than silently returning fixed glazing.
Before returning check: all labeled Aldoks positions have their own system; every opening symbol has an opening piece; double leaves are typed double/doubleSash; transoms remain fixed; only written outside dimensions become actual measurements.'''

ALLOWED = {'v','h','hv','hh','door','double','sash','doubleSash'}
def validate_result(result):
    if not isinstance(result, dict) or not isinstance(result.get('positions'), list):
        raise ValueError('positions listesi eksik')
    result['positions'] = result['positions'][:20]
    if not result['positions']: raise ValueError('Okunabilir poz bulunamadı')
    for i,pos in enumerate(result['positions'],1):
        if not isinstance(pos,dict): raise ValueError('Poz nesnesi geçersiz')
        for key in ('width','height'):
            value = pos.get(key)
            if value is None: continue
            if isinstance(value,bool): raise ValueError('Ölçü değeri geçersiz')
            try: number=float(str(value).strip().replace(',','.'))
            except (ValueError,TypeError): raise ValueError('P'+str(i)+' dış ölçüsü sayısal değil')
            if not math.isfinite(number) or not 30 <= number <= 3000:
                raise ValueError('P'+str(i)+' dış ölçüsü aralık dışında')
            pos[key]=number
        if pos.get('system') not in ('c60','aldoks',None): pos['system'] = None
        if pos.get('system_series') not in ('tam','yarim',None): pos['system_series'] = None
        parts = pos.get('pieces') or []
        if not isinstance(parts, list): raise ValueError('Parça listesi geçersiz')
        for p in parts:
            if not isinstance(p,dict) or p.get('kind') not in ALLOWED:
                raise ValueError('P'+str(i)+' parça türü beklenen biçimde değil')
            for key in ('x','y','doorWidth','doorHeight'):
                if p.get(key) is not None:
                    try:p[key]=float(str(p[key]).strip().replace(',','.'))
                    except (ValueError,TypeError):raise ValueError('P'+str(i)+' parça ölçüsü sayısal değil')
                    if not math.isfinite(p[key]):raise ValueError('Parça ölçüsü sonlu değil')
            for key in ('bounds','opening'):
                if isinstance(p.get(key),dict):
                    try:p[key]={k:float(str(p[key][k]).replace(',','.')) for k in ('left','right','top','bottom')}
                    except (KeyError,TypeError,ValueError):raise ValueError('P'+str(i)+' kanat sınırları eksik')
                    if not all(math.isfinite(v) for v in p[key].values()):raise ValueError('Kanat sınırları sonlu değil')
        pos['pieces'] = parts[:40]
    return result

def analysis_error(exc):
    # Do not log request bodies, image contents, credentials or raw exception messages.
    name=type(exc).__name__;status=getattr(exc,'status_code',None)
    code=getattr(exc,'code',None)
    known_codes={'insufficient_quota','rate_limit_exceeded','invalid_api_key','model_not_found','context_length_exceeded'}
    if code not in known_codes:code='unspecified'
    logger.error('OZC analysis failure type=%s upstream_status=%s code=%s',name,status,code)
    if isinstance(exc,json.JSONDecodeError):return 'AI_JSON: Model geçerli JSON döndürmedi. Tekrar deneyin.'
    if isinstance(exc,ValueError):return 'AI_FORMAT: '+str(exc)
    if status==401:return 'OPENAI_AUTH: Render OPENAI_API_KEY değeri OpenAI tarafından reddedildi.'
    if status==429:
        return 'OPENAI_QUOTA: API bakiyesi/kotası yetersiz.' if code=='insufficient_quota' else 'OPENAI_RATE: API kullanım sınırına ulaşıldı. Biraz sonra deneyin.'
    if status==404:return 'OPENAI_MODEL: Render OPENAI_MODEL değeri için model bulunamadı veya erişim yok.'
    if status==400:return 'OPENAI_REQUEST: Model veya görsel isteği reddedildi; OPENAI_MODEL ayarını kontrol edin.'
    if name=='APITimeoutError':return 'OPENAI_TIMEOUT: Görsel analizi zaman aşımına uğradı.'
    if name=='APIConnectionError':return 'OPENAI_CONNECTION: Sunucu OpenAI bağlantısını kuramadı.'
    return 'AI_INTERNAL: Analiz tamamlanamadı. Render Logs içindeki OZC analysis failure satırını paylaşın.'

@app.get('/health')
def health():
    return {'status':'ok','analyzer_version':VERSION,'diagnostics_version':'v123','capabilities':['opening_symbols','per_position_system','schematic_bounds']}

@app.post('/analyze')
async def analyze(image: UploadFile = File(...), x_ozc_access_key: str | None = Header(default=None)):
    access_key = os.getenv('OZC_ACCESS_KEY')
    if not access_key: raise HTTPException(503, 'Access control not configured')
    if not x_ozc_access_key or not secrets.compare_digest(x_ozc_access_key, access_key):
        raise HTTPException(401, 'Unauthorized')
    key = os.getenv('OPENAI_API_KEY')
    if not key: raise HTTPException(503, 'OPENAI_API_KEY not configured')
    if image.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(415, 'JPEG, PNG or WEBP required')
    content = await image.read(10*1024*1024+1)
    if len(content)>10*1024*1024: raise HTTPException(413, 'Maximum image size 10MB')
    url = 'data:'+image.content_type+';base64,'+base64.b64encode(content).decode()
    try:
        async with AsyncOpenAI(api_key=key, timeout=75, max_retries=0) as client:
            response = await client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL','gpt-4.1-mini'),
                response_format={'type':'json_object'}, temperature=0,max_tokens=6000,
                messages=[{'role':'system','content':INSTRUCTIONS},
                          {'role':'user','content':[{'type':'text','text':'Read every position, including opening symbols, binili kanat and local Aldoks labels. Return the complete JSON schema.'},
                                                    {'type':'image_url','image_url':{'url':url,'detail':'high'}}]}])
        if response.choices[0].finish_reason != 'stop': raise ValueError('Yanıt tamamlanmadı; görseli daha az poz içerecek şekilde bölüp deneyin')
        result = validate_result(json.loads(response.choices[0].message.content))
        return {'result':result,'analyzer_version':VERSION}
    except Exception as exc:
        raise HTTPException(502, analysis_error(exc))
