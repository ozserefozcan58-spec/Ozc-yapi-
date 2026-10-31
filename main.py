import os, base64, json, secrets, math, logging, unicodedata
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
An unmarked rectangular panel is FIXED glazing. The generic word window/pencere never means operable sash. Short diagonal measurement ticks across outer edges or across a fixed mullion are dimension marks, NEVER sash symbols. Only clear chevrons/X INSIDE the glazing, made of long lines meeting within the pane, are opening evidence. A vertical divider alone does not indicate a sash. Return opening_symbol=single_chevron/double_chevron/binili_symbol for every operable opening. Use no opening pieces for panels without interior opening marks. For handwritten corrected heights, use the uncrossed latest value; different heights on different positions are not contradictions.
First trace the structural grid before looking for openings. Outside arrows, leaders, curved arrows pointing at a record, dimension numerals and notebook grid lines are NEVER opening evidence. A horizontal line spanning the frame is a fixed horizontal mullion even when it crosses a vertical mullion. Keep BOTH records at their intersection. Never replace this grid with leaves based on the number, width or height of its rectangular panels.
For a written record height measured from the bottom, set h.y=(outside_height-written_height_from_bottom)/outside_height, not written_height_from_bottom/outside_height. Example: outside 300 x 270, one central vertical line, one full horizontal line labeled 200 from below, caption 'dar seri aldoks', and NO interior chevrons: system=aldoks, system_series=yarim, bottom_profile=kasa, pieces=[{kind:v,x:0.5,y:0.5},{kind:h,x:0.5,y:0.2592592593}], records=1. This is FOUR FIXED panes and ZERO openings; the 200 leader outside the left frame is only a dimension arrow. Apply the same reasoning to other dimensions, not a hardcoded size match.
User drawing convention: one chevron/triangle identifies one operable leaf (door or sash). Opposing chevrons or an X inside an operable opening denote BINILI double leaves: doubleSash for binili kanat, double for binili kapi. This is not a fixed-glass mullion. Show two leaves with no fixed middle mullion. A rectangle above leaves is fixed transom glass; do not turn the entire opening into a full-height sash. A neighboring rectangle is fixed glazing. Crossed-out text/dimensions are corrections, not opening symbols. 'Sonra imalat' is a production note, not a reason to discard an opening.
Bottom profile is INDEPENDENT of system. User drawing convention: a narrow band at the very BOTTOM of a fixed panel bounded by two parallel horizontal lines denotes ETEK (12 cm), even for ALDOKS. Set bottom_profile=etek for that position. Do not convert the band to a horizontal mullion, and do not default all Aldoks bottoms to kasa. A line above a door/leaf is a transom record, not an etek. A band inside a leaf is its leaf-bottom etek, not a fixed outer threshold. Use bottom_profile=kasa only for clearly indicated frame bottom; otherwise null.
Read labels BELOW the drawings and in notebook captions, not just above them. 'ALDOKS YARIM SERİ', 'yarım', 'yarim seri' and 'dar seri' all require system=aldoks, system_series=yarim; 'tam seri' and 'orta seri' require system_series=tam. Do not lose this explicit word when extracting dimensions. A caption beneath one drawing applies locally; a clearly common caption beneath a group applies to that group, never to unrelated C60 positions.
Respect labels local to each position: Aldoks written above a WC door applies to that door only. C60/RW60 and Aldoks must never be merged into one system for all positions. Labels may apply to neighboring similar doors. If Aldoks series (tam/yarim) is absent, report system=aldoks and system_series=null, plus a Turkish warning. Do not guess profile series.
Top-level keys: customer, project, system (c60/aldoks/null), glass, positions, warnings (Turkish strings).
Do not emit fixed glass, outer frames, sills or glazing beads in pieces; they are implicit. Only mullions and operable openings belong in pieces.
Each position: bottom_profile (etek/kasa/null), width, height, system (c60/aldoks/null), system_series (tam/yarim/null), records, pieces, warnings, note.
width/height must come from clearly written outside dimensions, never proportions. Use null for unreadable dimensions and warn; retain original visual order.
records counts fixed vertical mullions only, NOT sash stiles, double-leaf meeting stiles, frames or dimension ticks.
pieces contains all visible fixed mullions AND all openings. Allowed kind: v,h,hv,hh,door,double,sash,doubleSash. Normalized x/y are visual layout coordinates (0..1) only. Opening pieces require bounds={left,right,top,bottom} describing the visible opening within the outer frame (0..1), align=left/center/right, side=left/right, mainSide=right unless explicitly indicated. doorWidth/doorHeight must be explicit written internal opening dimensions, otherwise null. Flag schematic=true when bounds are estimated from sketch proportions. These proportions are for schematic display only, never manufacturing measurements. Never invent 90/180/220 dimensions. v/h pieces use x/y; hh also includes doorSpan=[left,right]. Do not duplicate an opening edge if the shared fixed mullion is already represented. If an opening symbol exists, pieces MUST NOT be empty. If type is uncertain, preserve your best structural classification and explain in Turkish warnings, rather than silently returning fixed glazing.
Before returning check: all labeled Aldoks positions have their own system; every opening symbol has an opening piece; double leaves are typed double/doubleSash; transoms remain fixed; only written outside dimensions become actual measurements.'''

ALLOWED = {'v','h','hv','hh','door','double','sash','doubleSash'}
def canonical_piece(piece):
    if not isinstance(piece,dict): raise ValueError('Parça nesnesi geçersiz')
    raw=piece.get('kind') or piece.get('type') or piece.get('piece_type') or ''
    token=''.join(ch for ch in unicodedata.normalize('NFKD',str(raw).lower().replace('ı','i')) if ch.isalnum() and not unicodedata.combining(ch))
    aliases={
        'v':'v','vertical':'v','verticalmullion':'v','verticalrecord':'v','mullion':'v','dikey':'v','dikeykayit':'v',
        'h':'h','horizontal':'h','horizontalmullion':'h','horizontalrecord':'h','transom':'h','yatay':'h','yataykayit':'h',
        'hv':'hv','halfvertical':'hv','halfverticalmullion':'hv','yarimdikeykayit':'hv',
        'hh':'hh','halfhorizontal':'hh','halfhorizontalmullion':'hh','yarimyataykayit':'hh',
        'door':'door','singledoor':'door','tek kapi':'door','tekkapi':'door','kapi':'door',
        'double':'double','binilidoor':'double','binilidoubledoor':'double','doubledoor':'double','doubleleafdoor':'double','binilikapi':'double','ciftkapi':'double',
        'sash':'sash','singleleaf':'sash','casement':'sash','kanat':'sash','tekkanat':'sash',
        'binilisash':'doubleSash','biniliwindow':'doubleSash','binilidoublesash':'doubleSash','biniliciftkanat':'doubleSash','doublecasement':'doubleSash','doublesash':'doubleSash','doublewindow':'doubleSash','doubleleafwindow':'doubleSash','doubleleaf':'doubleSash','binili':'doubleSash','binilikanat':'doubleSash','ciftkanat':'doubleSash',
    }
    static={'fixed','fixedglass','fixedpanel','fixedglazing','fixedwindow','glass','glazing','sabit','sabitcam','sabitpanel',
            'frame','outerframe','kasa','sill','bottomsill','etek','bottomprofile','bead','glazingbead','cita'}
    if token in static:return None
    if token in {'window','singlewindow','panel','pencere'}:
        # A whole window/panel is not evidence of an operable leaf.
        subtype=piece.get('opening_type') or piece.get('sash_type')
        if subtype and str(subtype).lower() not in {'window','singlewindow','panel','pencere'}:
            return canonical_piece({**piece,'kind':subtype})
        symbol=piece.get('opening_symbol')
        if symbol in {'double_chevron','binili_symbol'}:
            return {**piece,'kind':'doubleSash'}
        if symbol=='single_chevron':
            return {**piece,'kind':'sash'}
        return None
    kind=aliases.get(token)
    if kind is None:
        # A generic record needs its explicit orientation; never assume an opening is fixed.
        orientation=str(piece.get('orientation') or '').lower()
        if token in ('record','kayit') and orientation in ('vertical','dikey'):kind='v'
        elif token in ('record','kayit') and orientation in ('horizontal','yatay'):kind='h'
        else:raise ValueError('Tanınmayan parça türü: '+(token[:40] or 'tür bilgisi eksik'))
    out=dict(piece);out['kind']=kind
    return out

def canonical_series(value):
    token=''.join(ch for ch in unicodedata.normalize('NFKD',str(value or '').lower().replace('ı','i')) if ch.isalnum() and not unicodedata.combining(ch))
    if token in {'yarim','yarimseri','dar','darseri','aldoksyarim','aldoksyarimseri','aldoksdarseri'}:return 'yarim'
    if token in {'tam','tamseri','orta','ortaseri','aldokstam','aldokstamseri','aldoksortaseri'}:return 'tam'
    return None

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
        local_system=str(pos.get('system') or '').lower()
        series=canonical_series(pos.get('system_series') or pos.get('series')) or canonical_series(pos.get('system'))
        if 'aldoks' in local_system or series:pos['system']='aldoks'
        elif pos.get('system') not in ('c60',None):pos['system']=None
        if not series and pos.get('system')=='aldoks':series=canonical_series(result.get('system_series'))
        pos['system_series']=series
        parts = pos.get('pieces') or []
        if not isinstance(parts, list): raise ValueError('Parça listesi geçersiz')
        bottom=pos.get('bottom_profile',pos.get('bottom'))
        bottom_aliases={'etek':'etek','sill':'etek','bottom_sill':'etek','12 cm etek':'etek','kasa':'kasa','frame':'kasa'}
        pos['bottom_profile']=bottom_aliases.get(str(bottom).strip().lower()) if bottom is not None else None
        normalized=[]
        generic_count=0
        for raw_piece in parts:
            try:p=canonical_piece(raw_piece)
            except ValueError as exc:raise ValueError('P'+str(i)+' '+str(exc))
            if p is None:
                raw_kind=str(raw_piece.get('kind') or raw_piece.get('type') or '').lower() if isinstance(raw_piece,dict) else ''
                if raw_kind in {'etek','sill','bottom_sill','bottomsill'} and raw_piece.get('scope') not in {'leaf','sash','door'}:
                    pos['bottom_profile']='etek'
                if isinstance(raw_piece,dict) and str(raw_piece.get('kind') or raw_piece.get('type') or '').lower() in {'window','singlewindow','panel','pencere'}:generic_count+=1
                continue
            if p['kind'] in {'door','double','sash','doubleSash'}:
                symbol=str(p.get('opening_symbol') or '').strip().lower()
                if symbol not in {'single_chevron','double_chevron','binili_symbol'}:
                    raise ValueError('P'+str(i)+' kanat/kapı için iç açılım işareti doğrulanamadı; ölçü oku kanat sayılamaz')
            normalized.append(p)
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
        pos['pieces'] = normalized[:40]
        if generic_count:
            warnings=pos.get('warnings')
            if not isinstance(warnings,list):warnings=[]
            warnings.append('Genel pencere türü açılır kanat sayılmadı; açılım işaretlerini önizlemede kontrol edin.')
            pos['warnings']=warnings
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
    return {'status':'ok','analyzer_version':VERSION,'diagnostics_version':'v131','capabilities':['opening_symbols','per_position_system','schematic_bounds']}

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
