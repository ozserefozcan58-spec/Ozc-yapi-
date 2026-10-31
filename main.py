import os, base64, json, secrets, math, logging, unicodedata
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

VERSION = 'ozc-v154-leaf-bay-index'
logger = logging.getLogger('uvicorn.error')
app = FastAPI(title='OZC Drawing Analyzer', version=VERSION)
origins = [x.strip() for x in os.getenv('ALLOWED_ORIGINS', '').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['POST', 'GET'], allow_headers=['*'])

INSTRUCTIONS = '''You read Turkish hand-drawn aluminium joinery sheets. Return JSON only.
Read ALL positions in visual order, left to right and top to bottom. Ignore material stock totals as positions.
Extract THREE separate things for EVERY position: (1) written outside width/height in cm, (2) frame/mullion layout, (3) opening symbols and system labels. Do not stop after measuring rectangles.
An unmarked rectangular panel is FIXED glazing. The generic word window/pencere never means operable sash. Short diagonal measurement ticks across outer edges or across a fixed mullion are dimension marks, NEVER sash symbols. Only clear chevrons/X INSIDE the glazing, made of long lines meeting within the pane, are opening evidence. A vertical divider alone does not indicate a sash. Return opening_symbol=single_chevron/double_chevron/binili_symbol for every operable opening. Use no opening pieces for panels without interior opening marks. For handwritten corrected heights, use the uncrossed latest value; different heights on different positions are not contradictions.
Mentally rotate sideways photographs until the handwritten captions and dimensions are upright before classifying openings. A single large triangle/chevron inside a lower side bay IS a real opening symbol even when it touches the outer edge; a fixed rectangle above it must remain a transom. For EVERY opening piece put opening_symbol directly on that piece (single_chevron, double_chevron or binili_symbol), never only in a note or position-level field.
Count the full-height interior vertical dividers across the TOP of the frame first. N dividers means N+1 structural bays. Do not force equal thirds or merge a blank fixed bay between two doors. Read bays left to right before assigning any openings. For EACH position include layout_audit={bay_count,vertical_record_count,bays:[{type:fixed/door/double/sash/doubleSash,has_transom:boolean}]}. Count door meeting stiles separately from fixed dividers. In layout_audit, bay_count and bays describe STRUCTURAL bays from the top edge ONLY, not individual glass panes, transom panes or moving leaves. One double door is ONE structural bay, not two. Two full-height fixed dividers around a central double door mean THREE bays: fixed / double / fixed. Three full-height dividers with a single door and a separate double door mean FOUR bays: fixed / door / fixed / double. Both chevrons inside one double-door opening belong to ONE double piece. A transom above each door adds ZERO structural bays and is a local hh record. Do not add a central v for the double-door meeting stile. Apply these rules by visible topology, never by matching outside dimensions. An alternating fixed / single door / fixed / double door drawing has FOUR bays and THREE full-height vertical mullions, with TWO SEPARATE short transoms, one above each door. A fixed bay between doors must remain intact. Do not bridge the two transoms through that fixed bay. This applies regardless of outside dimensions.
A chevron in a bay with NO bottom threshold, reaching the same ground line as the neighboring fixed etek, identifies a FLOOR DOOR rather than a window sash. Set floor_opening=true only when this ground opening is visually clear, false for windows with a fixed sill below. Use kind=door or double for floor openings and retain bounds for the actual full opening up to its transom. Do not invert top/bottom coordinates: y=0 at the top, y=1 at the ground. A transom near the upper quarter cannot have y near 0.66. For a labeled outside-to-record distance, emit position_from_left_cm on the v piece; preserve this written position instead of equal spacing. Exterior dimension arrows are measurements, not opening marks. Missing internal door dimensions stay null so the app uses standard doors.
First trace the structural grid before looking for openings. Outside arrows, leaders, curved arrows pointing at a record, dimension numerals and notebook grid lines are NEVER opening evidence. A horizontal line spanning the frame is a fixed horizontal mullion even when it crosses a vertical mullion. Keep BOTH records at their intersection. Never replace this grid with leaves based on the number, width or height of its rectangular panels.
For a written record height measured from the bottom, set h.y=(outside_height-written_height_from_bottom)/outside_height, not written_height_from_bottom/outside_height. Example: outside 300 x 270, one central vertical line, one full horizontal line labeled 200 from below, caption 'dar seri aldoks', and NO interior chevrons: system=aldoks, system_series=yarim, bottom_profile=kasa, pieces=[{kind:v,x:0.5,y:0.5},{kind:h,x:0.5,y:0.2592592593}], records=1. This is FOUR FIXED panes and ZERO openings; the 200 leader outside the left frame is only a dimension arrow. Apply the same reasoning to other dimensions, not a hardcoded size match.
User drawing convention: one chevron/triangle identifies one operable leaf (door or sash). Opposing chevrons or an X inside an operable opening denote BINILI double leaves: doubleSash for binili kanat, double for binili kapi. This is not a fixed-glass mullion. Show two leaves with no fixed middle mullion. A rectangle above leaves is fixed transom glass; do not turn the entire opening into a full-height sash. A neighboring rectangle is fixed glazing. Crossed-out text/dimensions are corrections, not opening symbols. 'Sonra imalat' is a production note, not a reason to discard an opening.
Bottom profile is INDEPENDENT of system. User drawing convention: a narrow band at the very BOTTOM of a fixed panel bounded by two parallel horizontal lines denotes ETEK (12 cm), even for ALDOKS. Set bottom_profile=etek for that position. Do not convert the band to a horizontal mullion, and do not default all Aldoks bottoms to kasa. A line above a door/leaf is a transom record, not an etek. A band inside a leaf is its leaf-bottom etek, not a fixed outer threshold. Use bottom_profile=kasa only for clearly indicated frame bottom; otherwise null.
Read labels BELOW the drawings and in notebook captions, not just above them. 'ALDOKS YARIM SERİ', 'yarım', 'yarim seri' and 'dar seri' all require system=aldoks, system_series=yarim; 'tam seri' and 'orta seri' require system_series=tam. Do not lose this explicit word when extracting dimensions. A caption beneath one drawing applies locally; a clearly common caption beneath a group applies to that group, never to unrelated C60 positions.
Respect labels local to each position: Aldoks written above a WC door applies to that door only. C60/RW60 and Aldoks must never be merged into one system for all positions. Labels may apply to neighboring similar doors. If Aldoks series (tam/yarim) is absent, report system=aldoks and system_series=null, plus a Turkish warning. Do not guess profile series.
Top-level keys: customer, project, system (c60/aldoks/null), glass, positions, warnings (Turkish strings).
Do not emit fixed glass, outer frames, sills or glazing beads in pieces; they are implicit. Only mullions and operable openings belong in pieces.
Each position: bottom_evidence (single_line/paired_lines/unclear), bottom_profile (etek/kasa/null), width, height, system (c60/aldoks/null), system_series (tam/yarim/null), records, pieces, warnings, note.
width/height must come from clearly written outside dimensions, never proportions. Use null for unreadable dimensions and warn; retain original visual order.
records counts fixed vertical mullions only, NOT sash stiles, double-leaf meeting stiles, frames or dimension ticks.
pieces contains all visible fixed mullions AND all openings. Allowed kind: v,h,hv,hh,door,double,sash,doubleSash. single_chevron/double_chevron/binili_symbol are opening_symbol values, NEVER kind values. Normalized x/y are visual layout coordinates (0..1) only. Opening pieces require bounds={left,right,top,bottom} describing the visible opening within the outer frame (0..1), align=left/center/right, side=left/right, mainSide=right unless explicitly indicated. doorWidth/doorHeight must be explicit written internal opening dimensions, otherwise null. Flag schematic=true when bounds are estimated from sketch proportions. These proportions are for schematic display only, never manufacturing measurements. For doors with missing written internal measurements return doorWidth=null and/or doorHeight=null. The application applies the user-approved defaults (single 90 x 220 cm, double 180 x 220 cm) only for those missing axes; never fill a missing door measurement from proportions. A marked opening reaching the floor is a door; a fixed rectangle above it remains a transom. Keep its left/right location. For sash/doubleSash without written dimensions retain null and schematic bounds; do not apply door defaults to windows. v/h pieces use x/y. h means FULL frame width: use it only for a line visibly crossing all bays. A line only above a side door MUST be hh with doorSpan=[left,right] limited to that side bay, never h. Set scope=full only for a confirmed full-width horizontal. hh also includes doorSpan=[left,right]. Do not duplicate an opening edge if the shared fixed mullion is already represented. If an opening symbol exists, pieces MUST NOT be empty. If type is uncertain, preserve your best structural classification and explain in Turkish warnings, rather than silently returning fixed glazing.
Mandatory visual evidence: set bottom_evidence separately for EVERY position AFTER tracing its own fixed pane bottom. single_line means exactly one hand-drawn bottom boundary and requires bottom_profile=kasa. paired_lines means TWO distinct hand-drawn, closely spaced parallel bottom strokes bounding a narrow band on at least one FIXED pane and requires bottom_profile=etek. Ignore printed notebook grid lines, paper edges, shadows, retraced thick strokes, upper transoms and bands inside moving leaves. Never copy bottom classification from a neighboring position. If strokes cannot be distinguished, use unclear and bottom_profile=null. Do not infer evidence from the system, dimensions or your previously selected bottom_profile.
Final bottom audit: independently inspect the bottom of EACH FIXED pane, including panes beside a door. If only the fixed side has a double horizontal bottom line and the door side has no fixed threshold, still set position bottom_profile=etek; the drawing engine cuts that etek at the door. Never require the band to run across the entire position. Example: caption tam seri aldoks, left fixed pane with paired bottom lines, right floor-reaching single chevron plus fixed rectangle above: one door, left etek, right transom, not four-side kasa.
Written record dimensions have priority over sketch proportions. Inspect dimension leaders BELOW each position, especially arrows from a written number to a vertical divider. A number pointing at the first vertical divider from the left outside edge is position_from_left_cm on that v piece, NOT doorWidth. Record dimensions refer to profile centers. Return null position_from_left_cm for unmeasured dividers; never invent measured offsets from proportions. The app will distribute remaining fixed glazing equally with door widths preserved. Inspect EACH opening separately for the second chevron, especially the rightmost bay; never reuse the single-leaf classification of another opening.
Opening evidence audit is REQUIRED on EVERY opening piece: leaf_count=1 or 2, floor_opening=true or false, fixed_sill_below=true or false, has_transom=true or false. Count the actual long chevrons independently of kind. Two opposing chevrons in ONE structural bay require leaf_count=2 and opening_symbol=double_chevron even if they do not meet as an X. An opening reaching the bottom with no horizontal fixed threshold requires floor_opening=true and fixed_sill_below=false; a sill only in the NEIGHBORING fixed bay does not make it a window. A rectangle above those chevrons requires has_transom=true and an hh above that bay. Do not use the top of the outer frame as the leaf top when a transom line exists. Missing door height stays null, so the app applies 220 cm. Classify each opening afresh from the photo, not from its previous kind.
Before returning check: all labeled Aldoks positions have their own system; every opening symbol has an opening piece; double leaves are typed double/doubleSash; transoms remain fixed; only written outside dimensions become actual measurements.'''

ALLOWED = {'v','h','hv','hh','door','double','sash','doubleSash'}
def canonical_piece(piece):
    if not isinstance(piece,dict): raise ValueError('Parça nesnesi geçersiz')
    raw=piece.get('kind') or piece.get('type') or piece.get('piece_type') or ''
    token=''.join(ch for ch in unicodedata.normalize('NFKD',str(raw).lower().replace('ı','i')) if ch.isalnum() and not unicodedata.combining(ch))
    # Some model responses put the opening symbol in kind instead of opening_symbol.
    symbol_types={'singlechevron':'single_chevron','doublechevron':'double_chevron','binilisymbol':'binili_symbol'}
    if token in symbol_types:
        symbol=symbol_types[token]
        bounds=piece.get('bounds') or piece.get('opening') or {}
        subtype=piece.get('opening_type') or piece.get('sash_type')
        if subtype and str(subtype).lower().replace('_','') not in symbol_types:
            return canonical_piece({**piece,'kind':subtype,'opening_symbol':symbol})
        bottom=bounds.get('bottom') if isinstance(bounds,dict) else None
        try: grounded=bottom is not None and float(str(bottom).replace(',','.'))>=.97
        except (ValueError,TypeError):grounded=False
        if bottom is None:
            raise ValueError('Açılım işareti var ancak kapı/kanat ayrımı için alt sınır eksik')
        kind=('door' if grounded else 'sash') if token=='singlechevron' else ('double' if grounded else 'doubleSash')
        return {**piece,'kind':kind,'opening_symbol':symbol}
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
    if piece.get('opening_symbol') in {'double_chevron','binili_symbol'}:
        if kind=='door':kind='double'
        elif kind=='sash':kind='doubleSash'
    if piece.get('floor_opening') is True:
        if kind=='sash':kind='door'
        elif kind=='doubleSash':kind='double'
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
        evidence=pos.get('bottom_evidence')
        if evidence not in {'single_line','paired_lines','unclear'}:evidence='unclear'
        pos['bottom_evidence']=evidence
        if evidence=='single_line':pos['bottom_profile']='kasa'
        elif evidence=='paired_lines':pos['bottom_profile']='etek'
        else:pos['bottom_profile']=None
        normalized=[]
        generic_count=0
        for raw_piece in parts:
            try:p=canonical_piece(raw_piece)
            except ValueError as exc:raise ValueError('P'+str(i)+' '+str(exc))
            if p is None:
                raw_kind=str(raw_piece.get('kind') or raw_piece.get('type') or '').lower() if isinstance(raw_piece,dict) else ''
                if raw_kind in {'etek','sill','bottom_sill','bottomsill'} and raw_piece.get('scope') not in {'leaf','sash','door'}:
                    if pos['bottom_evidence']=='paired_lines':pos['bottom_profile']='etek'
                if isinstance(raw_piece,dict) and str(raw_piece.get('kind') or raw_piece.get('type') or '').lower() in {'window','singlewindow','panel','pencere'}:generic_count+=1
                continue
            if p['kind'] in {'door','double','sash','doubleSash'}:
                # Apply explicit visual evidence, never height/proportion heuristics.
                is_double=p.get('leaf_count')==2 or p.get('opening_symbol') in {'double_chevron','binili_symbol'} or p['kind'] in {'double','doubleSash'}
                if p.get('floor_opening') is True and p.get('fixed_sill_below') is False:
                    p['kind']='double' if is_double else 'door'
                elif is_double:
                    p['kind']='double' if p['kind']=='door' else 'doubleSash' if p['kind']=='sash' else p['kind']
                if is_double:p['opening_symbol']='double_chevron'
                symbol=str(p.get('opening_symbol') or '').strip().lower()
                if symbol not in {'single_chevron','double_chevron','binili_symbol'}:
                    raise ValueError('P'+str(i)+' kanat/kapı için iç açılım işareti doğrulanamadı; ölçü oku kanat sayılamaz')
            if p['kind']=='v' and p.get('position_from_left_cm') is not None:
                try: offset=float(str(p['position_from_left_cm']).replace(',','.'))
                except (ValueError,TypeError):raise ValueError('Kayıt konumu sayısal değil')
                if not math.isfinite(offset) or offset<=0 or (pos.get('width') is not None and offset>=pos['width']):
                    raise ValueError('Kayıt konumu dış ölçünün dışında: P'+str(i)+', soldan konum='+str(offset)+' cm, dış en='+str(pos.get('width'))+' cm')
                p['position_from_left_cm']=offset
                if pos.get('width') is not None:p['x']=offset/pos['width']
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
        audit=pos.get('layout_audit')
        if isinstance(audit,dict) and isinstance(audit.get('bay_count'),int):
            actual=sum(p['kind']=='v' for p in normalized)
            if audit['bay_count']!=actual+1:
                raise ValueError('Bölme sayısı ile dikey kayıt sayısı uyuşmuyor: P'+str(i)+' bölme='+str(audit['bay_count'])+', dikey kayıt='+str(actual)+', beklenen bölme='+str(actual+1))
        # A numerically consistent audit can still erase a fixed bay. Check
        # every opening against the actual divider intervals and bay labels.
        if isinstance(audit,dict) and isinstance(audit.get('bays'),list):
            edges=[0.0]+sorted(p['x'] for p in normalized if p['kind']=='v' and p.get('x') is not None)+[1.0]
            bays=audit['bays']
            if len(bays)!=len(edges)-1:
                raise ValueError('Bölme yerleşimi tutarsız: P'+str(i)+' bölme listesi ile kayıt aralıkları farklı')
            # If the independently enumerated bay sequence has exactly one
            # opening per operable bay, match schematic openings by left-to-right
            # order. Their sketch bounds need not be precise measurement data.
            bay_indices=[j for j,b in enumerate(bays) if b.get('type') in {'door','double','sash','doubleSash'}]
            opening_parts=sorted([q for q in normalized if q['kind'] in {'door','double','sash','doubleSash'}],key=lambda q: ((q.get('bounds') or q.get('opening') or {}).get('left',q.get('x',0))+(q.get('bounds') or q.get('opening') or {}).get('right',q.get('x',0)))/2)
            if len(opening_parts)==len(bay_indices):
                for opening,j in zip(opening_parts,bay_indices):
                    box=opening.get('bounds') or opening.get('opening')
                    expected=bays[j].get('type')
                    same_family=(expected in {'door','double'} and opening['kind'] in {'door','double'}) or (expected in {'sash','doubleSash'} and opening['kind'] in {'sash','doubleSash'})
                    if opening.get('schematic') is True and box and same_family:
                        box['left']=edges[j];box['right']=edges[j+1]
                        opening['x']=(edges[j]+edges[j+1])/2
            for opening in normalized:
                if opening['kind'] not in {'door','double','sash','doubleSash'}:continue
                box=opening.get('bounds') or opening.get('opening')
                if not box:continue
                center=(box['left']+box['right'])/2
                bay=next((j for j in range(len(edges)-1) if edges[j]<=center<edges[j+1]),None)
                if bay is None:raise ValueError('Bölme yerleşimi tutarsız: P'+str(i)+' açılım çerçeve dışında')
                if bays[bay].get('type')=='fixed':
                    raise ValueError('Bölme yerleşimi tutarsız: P'+str(i)+' kapı sabit cam bölmesine konmuş; eksik dikey kayıt veya yanlış kapı sınırı')
                if box['left']<edges[bay]-.02 or box['right']>edges[bay+1]+.02:
                    expected_type=bays[bay].get('type')
                    same_family=(expected_type in {'door','double'} and opening['kind'] in {'door','double'}) or (expected_type in {'sash','doubleSash'} and opening['kind'] in {'sash','doubleSash'})
                    # Sketch proportions are not manufacturing measurements. A single
                    # opening with matching bay evidence belongs inside that bay.
                    occupants=[q for q in normalized if q['kind'] in {'door','double','sash','doubleSash'} and isinstance(q.get('bounds') or q.get('opening'),dict) and edges[bay]<=(lambda b:(b['left']+b['right'])/2)(q.get('bounds') or q.get('opening'))<edges[bay+1]]
                    if opening.get('schematic') is True and same_family and len(occupants)==1:
                        box['left']=edges[bay];box['right']=edges[bay+1]
                        opening['x']=(edges[bay]+edges[bay+1])/2
                        pos.setdefault('warnings',[])
                        if not isinstance(pos['warnings'],list):pos['warnings']=[]
                        pos['warnings'].append('Şematik açılım sınırı kendi yapısal bölmesine hizalandı; yazılı ölçüler korundu.')
                    else:
                        raise ValueError('Bölme yerleşimi tutarsız: P'+str(i)+' açılım sabit kayıt üzerinden komşu bölmeye taşıyor')
                expected=bays[bay].get('type')
                if expected in {'double','doubleSash'} and opening['kind'] in {'door','sash'}:
                    raise ValueError('Bölme yerleşimi tutarsız: P'+str(i)+' binili bölme tek kanat olarak aktarılmış')
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
    return {'status':'ok','analyzer_version':VERSION,'diagnostics_version':'v154','capabilities':['opening_symbols','per_position_system','schematic_bounds']}

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
    analysis_attempts=[]
    stage='initial_analysis'
    review_reason=None
    try:
        messages=[{'role':'system','content':INSTRUCTIONS},
                  {'role':'user','content':[{'type':'text','text':'Read every position, including opening symbols, binili kanat and local Aldoks labels. Return the complete JSON schema. Independently trace each fixed bottom and include bottom_evidence=single_line/paired_lines/unclear for EACH position. Every opening piece must include its own opening_symbol.'},
                                            {'type':'image_url','image_url':{'url':url,'detail':'high'}}]}]
        async with AsyncOpenAI(api_key=key, timeout=50, max_retries=0) as client:
            async def request_analysis(review=False):
                response = await client.chat.completions.create(
                    model=os.getenv('OPENAI_MODEL','gpt-4.1-mini'),
                    response_format={'type':'json_object'}, temperature=0,max_tokens=6000,
                    timeout=25 if review else 50, messages=messages)
                if response.choices[0].finish_reason != 'stop':
                    raise ValueError('Yanıt tamamlanmadı; görseli daha az poz içerecek şekilde bölüp deneyin')
                parsed=json.loads(response.choices[0].message.content)
                # Snapshot before validation, which normalizes the response in place.
                analysis_attempts.append({'stage':'review' if review else 'initial','result':json.loads(json.dumps({k:parsed.get(k) for k in ('system','glass','positions','warnings')}))})
                return parsed
            raw = await request_analysis()
            stage='initial_validation'
            review_reason=None
            try:
                result = validate_result(raw)
                if any(p.get('width') is None or p.get('height') is None for p in result['positions']):
                    review_reason='An outside width or height is missing. Re-read the written dimension ABOVE and BESIDE each position. Keep position order and preserve every readable position_from_left_cm; a missing outside width does NOT invalidate an independently written record offset. Do not substitute the record dimension for the outside width. Do not infer dimensions from proportions.'
                elif any(p.get('bottom_profile') is None for p in result['positions']):
                    review_reason='At least one bottom_profile is null. Inspect each fixed pane bottom carefully for the paired horizontal ETEK band, including a band only beside a door. Recheck all local transoms: a record above a side door is hh with doorSpan, not a full-width h.'
            except ValueError as exc:
                if not any(reason in str(exc) for reason in ('iç açılım işareti doğrulanamadı','Bölme sayısı ile dikey kayıt sayısı uyuşmuyor','Kayıt konumu dış ölçünün dışında','Bölme yerleşimi tutarsız')):raise
                review_reason=str(exc)+'. Recheck written record-distance leaders and their units against the ORIGINAL photo. position_from_left_cm must be an actual left-outside-edge to divider CENTER distance in CENTIMETERS, strictly within outside width. Do not confuse outside width, height, panel width, stock lengths, or a neighboring position dimension with this offset. Convert mm to cm only when the written unit confirms millimeters. If unreadable, return null and warn in Turkish; never clamp or invent an offset. Recount STRUCTURAL bays along the TOP edge, not glass panes or leaves. A double door counts as ONE bay and its meeting stile is NOT v. Keep every fixed bay between separate doors. Correct the inconsistent pieces or audit from the photo; never add a divider just to satisfy arithmetic.'
            if review_reason is None and any(sum(q['kind'] in {'door','double','sash','doubleSash'} for q in p['pieces'])>=2 and not isinstance(p.get('layout_audit'),dict) for p in result['positions']):
                review_reason='Multiple operable openings: independently count the top-edge vertical dividers and every fixed bay BETWEEN openings. Verify layout_audit, door versus window using ground openings, all written record distances, and separate hh transoms without crossing intervening fixed panes.'
            if review_reason is None and any(q.get('leaf_count') not in (1,2) or not isinstance(q.get('floor_opening'),bool) or not isinstance(q.get('fixed_sill_below'),bool) or not isinstance(q.get('has_transom'),bool) for p in result['positions'] for q in p['pieces'] if q['kind'] in {'door','double','sash','doubleSash'}):
                review_reason='Opening evidence fields missing. Independently inspect each opening for ONE versus TWO chevrons, whether it reaches the ground WITHOUT a fixed sill, and whether a fixed transom rectangle is above it. Read every below-frame written record-distance arrow and emit position_from_left_cm on its v piece. Independently count BOTH chevrons in every rightmost opening. Return leaf_count, floor_opening, fixed_sill_below and has_transom on EACH opening, with corrected kind, opening_symbol and local hh transoms. A pair of opposing chevrons in a ground opening is one double DOOR, not a doubleSash or single door.'
            if review_reason:
                # Start the review from the original photo, without the failed model
                # response: repeating it can anchor the recount to missing dividers.
                messages=messages[:2]
                review_reason+=' Never repair a contradiction by reducing the bay_count or deleting a fixed bay. An alternating fixed/single door/fixed/double door sequence has four structural bays, three v pieces, one door piece, one double piece and two local hh transoms. A written divider distance locates the divider, not an opening. Classify the rightmost opening by its OWN chevrons. First trace ALL full-height INTERNAL vertical strokes from the top frame down to the floor/sill. Each stroke between neighboring structural bays must appear exactly once as kind=v in pieces, including the side jambs separating a door bay from fixed glazing. Only the central moving-leaf meeting stile is excluded. Build layout_audit from that same traced list: vertical_record_count equals the v count and bay_count equals that count plus one. Do not count transom glass as a separate structural bay. Next trace local transoms, then classify openings. Preserve every written dimension and the full list of positions.'
                messages.append({'role':'user','content':[{'type':'text','text':
                    'Recheck the ORIGINAL PHOTO independently. '+review_reason+
                    ' Mentally rotate if needed. Return FULL corrected JSON with all dimensions, mullions, transoms, series and bottom_profile. Include opening_symbol on each real opening. Preserve real openings. Return bottom_evidence for each position: paired_lines requires etek, single_line requires kasa, unclear requires null. Count only hand-drawn strokes at that position fixed bottom, ignoring printed grid lines. A band only beside a door still counts. Do not copy evidence between positions. Do not invent evidence.'}]})
                stage='review_analysis'
                reviewed=await request_analysis(review=True)
                stage='review_validation'
                result = validate_result(reviewed)
            # Focused visual leaf count: do not ask the model to regenerate layout.
            targets=[]
            for pi,pos in enumerate(result['positions']):
                for oi,piece in enumerate(pos['pieces']):
                    if piece['kind'] in {'door','sash'}:
                        box=piece.get('bounds') or piece.get('opening') or {}
                        center=(box.get('left',0)+box.get('right',1))/2
                        bay_index=sum(q['kind']=='v' and q.get('x',0)<center for q in pos['pieces'])
                        targets.append({'position_index':pi,'piece_index':oi,'structural_bay_index':bay_index,'structural_bay_count':sum(q['kind']=='v' for q in pos['pieces'])+1})
            if targets:
                stage='leaf_count_audit'
                try:
                    audit_response=await client.chat.completions.create(
                        model=os.getenv('OPENAI_MODEL','gpt-4.1-mini'),
                        response_format={'type':'json_object'},temperature=0,max_tokens=1800,timeout=15,
                        messages=[{'role':'system','content':'Inspect ONLY the number of moving leaves in the specified hand-drawn openings. Return JSON {"openings":[{"position_index":0,"piece_index":0,"chevron_count":1,"both_chevrons_visible":false}]}. Position and structural_bay indices are ZERO-BASED. Locate positions top to bottom and structural bays left to right using the PHOTO, not normalized measurement coordinates. Inspect the WHOLE requested bay from its left divider to right divider; do not crop to one leaf. Two opposing long chevrons inside the SAME opening indicate TWO leaves even if they do not touch or form an X. Trace the entire left AND right half of each target, especially the rightmost opening. A short handle mark is not a chevron. Count only complete opening chevrons, not transom lines, dimension arrows or neighboring openings. Use chevron_count=null if unclear. Do not output dimensions, frame layout, systems or records.'},
                                  {'role':'user','content':[{'type':'text','text':'Independently inspect these target regions on the ORIGINAL photo. Locate targets by position order and structural bay order; numeric measurement proportions are irrelevant. '+json.dumps(targets)}, {'type':'image_url','image_url':{'url':url,'detail':'high'}}]}])
                    if audit_response.choices[0].finish_reason!='stop':raise ValueError('Kanat sayısı kontrol yanıtı tamamlanmadı')
                    leaf_audit=json.loads(audit_response.choices[0].message.content)
                    analysis_attempts.append({'stage':'leaf_count_audit','result':leaf_audit})
                    allowed={(t['position_index'],t['piece_index']) for t in targets}
                    entries=leaf_audit.get('openings',[])
                    counts={}
                    for entry in entries:
                        pair=(entry.get('position_index'),entry.get('piece_index'))
                        counts[pair]=counts.get(pair,0)+1
                    for entry in entries:
                        pair=(entry.get('position_index'),entry.get('piece_index'))
                        if pair not in allowed or counts[pair]!=1:continue
                        if entry.get('chevron_count')!=2 or entry.get('both_chevrons_visible') is not True:continue
                        pi,oi=pair;piece=result['positions'][pi]['pieces'][oi]
                        piece['kind']='double' if piece['kind']=='door' else 'doubleSash'
                        piece['leaf_count']=2;piece['opening_symbol']='double_chevron'
                        result['positions'][pi].setdefault('warnings',[]).append('İkinci görsel kontrolde iki açılım işareti doğrulandı; binili kanat olarak düzeltildi.')
                except Exception:
                    # Supplemental counting must not discard a valid whole drawing.
                    result.setdefault('warnings',[]).append('Ek kanat sayısı kontrolü tamamlanamadı; tek/binili seçimini önizlemede kontrol edin.')
        return {'result':result,'analyzer_version':VERSION,'diagnostics':{'attempts':[a for a in analysis_attempts if a['stage']=='leaf_count_audit']}}
    except Exception as exc:
        detail=analysis_error(exc)
        if isinstance(exc,(ValueError,json.JSONDecodeError)):
            return JSONResponse(status_code=502,content={'detail':detail,'analyzer_version':VERSION,'diagnostics':{'stage':stage,'review_reason':review_reason,'attempts':analysis_attempts}})
        raise HTTPException(502, detail)
