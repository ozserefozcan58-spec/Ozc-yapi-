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
User drawing convention: one chevron/triangle identifies one operable leaf (door or sash). Opposing chevrons or an X inside an operable opening denote BINILI double leaves: doubleSash for binili kanat, double for binili kapi. This is not a fixed-glass mullion. Show two leaves with no fixed middle mullion. A rectangle above leaves is fixed transom glass; do not turn the entire opening into a full-height sash. A neighboring rectangle is fixed glazing. Crossed-out text/dimensions are corrections, not opening symbols. 'Sonra imalat' is a production note, not a reason to discard an opening.
Respect labels local to each position: Aldoks written above a WC door applies to that door only. C60/RW60 and Aldoks must never be merged into one system for all positions. Labels may apply to neighboring similar doors. If Aldoks series (tam/yarim) is absent, report system=aldoks and system_series=null, plus a Turkish warning. Do not guess profile series.
Top-level keys: customer, project, system (c60/aldoks/null), glass, positions, warnings (Turkish strings).
Do not emit fixed glass, outer frames, sills or glazing beads in pieces; they are implicit. Only mullions and operable openings belong in pieces.
Each position: width, height, system (c60/aldoks/null), system_series (tam/yarim/null), records, pieces, warnings, note.
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
        'double':'double','doubledoor':'double','doubleleafdoor':'double','binilikapi':'double','ciftkapi':'double',
        'sash':'sash','window':'sash','singlewindow':'sash','singleleaf':'sash','casement':'sash','kanat':'sash','tekkanat':'sash',
        'doublesash':'doubleSash','doublewindow':'doubleSash','doubleleafwindow':'doubleSash','doubleleaf':'doubleSash','binili':'doubleSash','binilikanat':'doubleSash','ciftkanat':'doubleSash',
    }
    static={'fixed','fixedglass','fixedpanel','fixedglazing','fixedwindow','glass','glazing','sabit','sabitcam','sabitpanel',
            'frame','outerframe','kasa','sill','bottomsill','etek','bottomprofile','bead','glazingbead','cita'}
    if token in static:return None
    kind=aliases.get(token)
    if kind is None:
        # A generic record needs its explicit orientation; never assume an opening is fixed.
        orientation=str(piece.get('orientation') or '').lower()
        if token in ('record','kayit') and orientation in ('vertical','dikey'):kind='v'
        elif token in ('record','kayit') and orientation in ('horizontal','yatay'):kind='h'
        else:raise ValueError('Tanınmayan parça türü: '+(token[:40] or 'tür bilgisi eksik'))
    out=dict(piece);out['kind']=kind
    return out

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
        normalized=[]
        for raw_piece in parts:
            try:p=canonical_piece(raw_piece)
            except ValueError as exc:raise ValueError('P'+str(i)+' '+str(exc))
            if p is None:continue
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
    return result

def analysis_error(exc):
    # Do not log request bodies, image contents, credentials or raw exception messages.
    name=type(exc).__name__;status=getattr(exc,'status_code',None)
    code=getattr(exc,'code',None)
    known_codes={'insufficient_quota','rate_limit_exceeded','invalid_api_key','model_not_found','context_length_exceeded'}
