from __future__ import annotations
import os, sys, re
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DATA = ROOT / 'data' / 'customer_loyalty_synthetic_20k'
PROCESSED = DATA / 'processed'
SEG = PROCESSED / 'segmentation_outputs'
RAG_ROOT = ROOT / 'nexora_rag_llm'
sys.path.insert(0, str(RAG_ROOT))

app = FastAPI(title='Nexora Bank | Internal Customer Engagement Assistant', version='1.0.0')

SEGMENT_MAP = {
    0: ('MODERATE', 'Moderate', 'Regular but moderate activity'),
    1: ('DORMANT', 'Dormant', 'Very low observed activity'),
    2: ('CONSISTENT', 'Consistent / loyal', 'High activity and engagement'),
    3: ('DECLINING_ACTIVITY', 'Declining activity', 'Lower activity and higher transaction recency'),
}

def load_csv(name, processed=False):
    path = (PROCESSED if processed else DATA) / name
    if not path.exists():
        raise RuntimeError(f'Missing required dataset file: {path}')
    return pd.read_csv(path)


# Load the exact fitted K-Means model and scaler used to create the supplied
# segmentation artefacts. Customer IDs are looked up in the profile table;
# their feature vector is rebuilt and passed through scaler.transform + predict.
@lru_cache(maxsize=1)
def segmentation_artifacts():
    import joblib
    model_path = SEG / 'kmeans_model.joblib'
    scaler_path = SEG / 'standard_scaler.joblib'
    metadata_path = SEG / 'model_metadata.json'
    if not all(p.exists() for p in (model_path, scaler_path, metadata_path)):
        raise RuntimeError('K-Means model, scaler or model metadata is missing.')
    return joblib.load(model_path), joblib.load(scaler_path), __import__('json').loads(metadata_path.read_text())

def customer_feature_vector(row, feature_columns):
    # Mirrors the feature engineering used for the saved training matrix.
    import numpy as np
    def num(key):
        value = row.get(key)
        return 0.0 if value is None or pd.isna(value) else float(value)
    def log1p(key):
        return float(np.log1p(max(0.0, num(key))))
    observed = max(1.0, num('observed_months'))
    tx = num('successful_transaction_count')
    login, sessions = num('avg_login_count'), num('avg_app_sessions')
    values = {
        'tenure_months': num('tenure_months'),
        'avg_active_days': num('avg_active_days'),
        'avg_login_count': log1p('avg_login_count'),
        'avg_app_sessions': log1p('avg_app_sessions'),
        'avg_service_usage': log1p('avg_service_usage'),
        'avg_engagement_score': num('avg_engagement_score'),
        'latest_engagement_score': num('latest_engagement_score'),
        'successful_transaction_count': log1p('successful_transaction_count'),
        'total_transaction_value_inr': log1p('total_transaction_value_inr'),
        'avg_transaction_value_inr': log1p('avg_transaction_value_inr'),
        'median_transaction_value_inr': log1p('median_transaction_value_inr'),
        'transaction_recency_days': num('transaction_recency_days'),
        'campaigns_received': log1p('campaigns_received'),
        'campaign_acceptance_rate': num('campaign_acceptance_rate'),
        'active_day_rate': num('avg_active_days') / 30.0,
        'sessions_per_login': sessions / login if login > 0 else 0.0,
        'transactions_per_observed_month': tx / observed,
        'spend_per_transaction_inr': num('total_transaction_value_inr') / tx if tx > 0 else 0.0,
    }
    return pd.DataFrame([[values.get(col, 0.0) for col in feature_columns]], columns=feature_columns)

def predict_customer_segment(profile_row):
    model, scaler, metadata = segmentation_artifacts()
    columns = metadata['feature_columns']
    features = customer_feature_vector(profile_row, columns)
    cluster = int(model.predict(scaler.transform(features))[0])
    return cluster

@lru_cache(maxsize=1)
def tables():
    profiles = load_csv('customer_profile.csv', True)
    assignments = pd.read_csv(SEG / 'customer_cluster_assignments.csv')
    profiles = profiles.merge(assignments, on='customer_id', how='left', validate='one_to_one')
    return {
        'profiles': profiles,
        'transactions': load_csv('transactions.csv'),
        'billing': load_csv('billing_history.csv'),
        'communications': load_csv('communication_history.csv'),
        'interactions': load_csv('campaign_interactions.csv'),
        'offers': load_csv('approved_offers.csv'),
        'campaigns': load_csv('campaign_catalogue.csv'),
        'cluster_summary': pd.read_csv(SEG / 'cluster_summary.csv'),
    }

def clean(v):
    if v is None: return None
    try:
        if pd.isna(v): return None
    except (TypeError, ValueError): pass
    if hasattr(v, 'item'): return v.item()
    if isinstance(v, (pd.Timestamp, datetime)): return v.isoformat()
    return v

def row_customer(customer_id: str):
    df = tables()['profiles']
    match = df[df.customer_id.astype(str).str.upper() == customer_id.strip().upper()]
    if match.empty: raise HTTPException(404, f'Customer {customer_id} not found')
    r = match.iloc[0].to_dict()
    # Inference is performed for each requested customer; the stored assignment
    # is retained only as a parity/audit reference, never as the prediction.
    stored_cluster = int(r.get('cluster', -1)) if pd.notna(r.get('cluster')) else -1
    cluster = predict_customer_segment(r)
    key, label, desc = SEGMENT_MAP.get(cluster, ('UNMAPPED', 'Needs review', 'Cluster mapping unavailable'))
    facts = {k: clean(r.get(k)) for k in r}
    facts.update({'cluster': cluster, 'stored_cluster_reference': stored_cluster,
                  'prediction_matches_reference': cluster == stored_cluster,
                  'segment': key, 'segment_label': label, 'segment_description': desc,
                  'segmentation_method': 'K-Means model inference using saved StandardScaler + feature engineering'})
    return facts

def category_from_input(text: str, chosen: str = 'auto'):
    if chosen and chosen != 'auto': return chosen
    t = (text or '').lower()
    patterns = [
      ('billing_reminder', r'overdue|late payment|payment due|bill|billing|repayment|installment|emi'),
      ('fee_waiver', r'fee waiver|waive fee|annual fee|maintenance fee|service charge|fee reversal'),
      ('cashback', r'cashback|cash back|retail spend'),
      ('bonus_points', r'bonus point|loyalty point|points reward|reward points'),
      ('offer_recommendation', r'offer|campaign|promotion|discount|cross.sell'),
      ('new_customer_onboarding', r'new customer|onboard|welcome|first use|activation'),
      ('re_engagement', r're.engage|dormant|inactive|win.back|low activity'),
      ('service', r'complaint|support|service request|account query|statement'),
    ]
    for name, pat in patterns:
        if re.search(pat, t): return name
    return 'general_engagement'

def history_for(customer_id):
    t = tables()
    return {
      'transactions': t['transactions'][t['transactions'].customer_id == customer_id],
      'billing': t['billing'][t['billing'].customer_id == customer_id],
      'communications': t['communications'][t['communications'].customer_id == customer_id],
      'interactions': t['interactions'][t['interactions'].customer_id == customer_id],
    }

def deterministic_checks(customer: dict, category: str, channel: str = 'email'):
    cid = customer['customer_id']; h = history_for(cid); now = pd.Timestamp.now().normalize()
    cons = customer.get('marketing_consent')
    consent_ok = bool(cons) if cons is not None else False
    comm = h['communications'].copy()
    if not comm.empty:
        comm['communication_date'] = pd.to_datetime(comm['communication_date'], errors='coerce')
        recent7 = comm[(comm.communication_date >= now - pd.Timedelta(days=7)) & (comm.communication_purpose.astype(str).str.lower().str.contains('promo|marketing|campaign|engage', regex=True))]
        monthly = comm[(comm.communication_date.dt.to_period('M') == now.to_period('M')) & (comm.communication_purpose.astype(str).str.lower().str.contains('promo|marketing|campaign|engage', regex=True))]
        recent7_count, monthly_count = len(recent7), len(monthly)
    else: recent7_count, monthly_count = 0, 0
    cap = int(customer.get('monthly_contact_cap') or 0)
    overdue = int(customer.get('overdue_bill_count') or 0)
    billing_records = h['billing']
    latest_bill = billing_records.sort_values('billing_month').tail(1).to_dict('records') if not billing_records.empty else []
    promotional = category not in ('billing_reminder', 'service')
    checks = {
      'customer_id': cid,
      'segment': customer['segment'],
      'marketing_consent': cons,
      'channel_requested': channel,
      'promotional_contact': promotional,
      'consent_check': 'PASS' if (not promotional or consent_ok) else 'FAIL',
      'profile_monthly_contact_cap': cap,
      'promotional_contacts_last_7_days_observed': recent7_count,
      'promotional_contacts_this_month_observed': monthly_count,
      'overdue_bill_count': overdue,
      'billing_record_count': len(billing_records),
      'latest_billing_record': {k: clean(v) for k,v in latest_bill[0].items()} if latest_bill else None,
      'offer_eligibility': 'NOT_EVALUATED',
      'reason': 'Offer eligibility requires campaign enrolment, exact product/account, qualifying transactions and benefit-ledger evidence; this dataset does not provide all authoritative fields.'
    }
    if promotional and not consent_ok:
        checks['overall_gate'] = 'NEEDS_REVIEW'
        checks['block_reason'] = 'No current recorded marketing consent in the customer profile.'
    elif category == 'billing_reminder' and (not latest_bill or overdue <= 0):
        checks['overall_gate'] = 'NEEDS_REVIEW'
        checks['block_reason'] = 'No verified overdue bill is available to support a payment reminder.'
    elif category == 'billing_reminder':
        checks['overall_gate'] = 'REVIEW_REQUIRED'
        checks['block_reason'] = 'A billing record exists; verify latest amount, due date, payment status and dispute/hardship status before approval.'
    else:
        checks['overall_gate'] = 'DRAFT_ALLOWED_SUBJECT_TO_REVIEW'
    return checks

@app.get('/api/health')
def health():
    return {'status':'ok', 'dataset':'synthetic hackathon data', 'customer_count':len(tables()['profiles']), 'rag_available':(RAG_ROOT/'src'/'nexora_rag').exists(), 'llm_provider':os.getenv('LLM_PROVIDER','gemini')}

@app.get('/api/dashboard')
def dashboard():
    t=tables(); df=t['profiles']; counts=df.cluster.value_counts().to_dict(); total=len(df)
    segments=[]
    for cluster,(key,label,desc) in SEGMENT_MAP.items():
        n=int(counts.get(cluster,0)); segments.append({'id':cluster,'key':key,'label':label,'description':desc,'count':n,'percent':round(100*n/total,2) if total else 0})
    acceptance=pd.to_numeric(df.campaign_acceptance_rate,errors='coerce').mean()
    return {'total_customers':total,'consistent_customers':int(counts.get(2,0)),'attention_customers':int(counts.get(1,0)+counts.get(3,0)),'campaign_acceptance_rate':clean(acceptance),'active_offers':int(t['offers'].active.astype(str).str.lower().isin(['true','1']).sum()),'approved_campaigns':int(t['campaigns'].approved.astype(str).str.lower().isin(['true','1']).sum()),'segments':segments}

@app.get('/api/customers')
def list_customers(q:str='', segment:str='', limit:int=25):
    df=tables()['profiles'].copy()
    if q:
        df=df[df.customer_id.astype(str).str.contains(q,case=False,na=False)]
    if segment:
        wanted=segment.upper()
        ids=[k for k,v in SEGMENT_MAP.items() if v[0]==wanted]
        df=df[df.cluster.isin(ids)]
    cols=['customer_id','primary_product','customer_status','cluster','avg_engagement_score','transaction_recency_days','overdue_bill_count','campaign_acceptance_rate']
    out=[]
    for row in df.head(max(1,min(limit,100))).to_dict('records'):
        cid=int(row.get('cluster',-1)); row['segment']=SEGMENT_MAP.get(cid,('UNKNOWN','Unknown',''))[0]
        out.append({k:clean(v) for k,v in row.items() if k in cols+['segment']})
    return {'items':out,'total':int(len(df))}

@app.get('/api/customers/{customer_id}')
def get_customer(customer_id:str):
    customer=row_customer(customer_id)
    h=history_for(customer['customer_id'])
    customer['history_summary']={k:len(v) for k,v in h.items()}
    customer['recent_transactions']=[{k:clean(v) for k,v in r.items()} for r in h['transactions'].sort_values('transaction_date').tail(5).to_dict('records')]
    customer['recent_billing']=[{k:clean(v) for k,v in r.items()} for r in h['billing'].sort_values('billing_month').tail(5).to_dict('records')]
    customer['recent_communications']=[{k:clean(v) for k,v in r.items()} for r in h['communications'].sort_values('communication_date').tail(5).to_dict('records')]
    customer['recent_campaign_interactions']=[{k:clean(v) for k,v in r.items()} for r in h['interactions'].tail(5).to_dict('records')]
    offers=tables()['offers'].copy()
    offers['active']=offers['active'].astype(str).str.lower().isin(['true','1'])
    customer['active_offer_catalogue']=[{k:clean(v) for k,v in row.items()} for row in offers[offers.active].to_dict('records')]
    customer['offer_eligibility_notice']='Offers shown are catalogue candidates only. Customer-specific eligibility is not established by this dataset/API.'
    return customer

class ClassifyRequest(BaseModel):
    customer_id: str
    input_text: str = ''
    category: str = 'auto'
    channel: str = 'email'

@app.post('/api/classify')
def classify(req:ClassifyRequest):
    customer=row_customer(req.customer_id)
    category=category_from_input(req.input_text,req.category)
    checks=deterministic_checks(customer,category,req.channel)
    h=history_for(customer['customer_id'])
    offers=tables()['offers'].copy()
    offers['active']=offers['active'].astype(str).str.lower().isin(['true','1'])
    customer_snapshot={
      'recent_transactions':[{k:clean(v) for k,v in r.items()} for r in h['transactions'].sort_values('transaction_date').tail(5).to_dict('records')],
      'recent_billing':[{k:clean(v) for k,v in r.items()} for r in h['billing'].sort_values('billing_month').tail(5).to_dict('records')],
      'recent_communications':[{k:clean(v) for k,v in r.items()} for r in h['communications'].sort_values('communication_date').tail(5).to_dict('records')],
      'recent_campaign_interactions':[{k:clean(v) for k,v in r.items()} for r in h['interactions'].tail(5).to_dict('records')],
      'active_offer_catalogue':[{k:clean(v) for k,v in r.items()} for r in offers[offers.active].to_dict('records')]
    }
    return {'customer':customer,'customer_snapshot':customer_snapshot,
            'classified_category':category,'category_label':category.replace('_',' ').title(),
            'checks':checks,'human_review_required':True}

class DraftRequest(BaseModel):
    customer_id:str
    input_text:str=Field(default='Prepare an appropriate customer communication draft.')
    category:str='auto'
    channel:str='email'
    objective:str=''

@app.post('/api/draft')
def create_draft(req:DraftRequest):
    customer=row_customer(req.customer_id)
    category=category_from_input(req.input_text or req.objective,req.category)
    checks=deterministic_checks(customer,category,req.channel)
    if checks['overall_gate']=='NEEDS_REVIEW' and checks.get('consent_check')=='FAIL':
        return {'status':'REVIEW_REQUIRED','category':category,'customer':customer,'checks':checks,'draft':None,'warnings':[checks.get('block_reason')], 'retrieved_sources':[]}
    if category=='billing_reminder' and not checks.get('latest_billing_record'):
        return {'status':'REVIEW_REQUIRED','category':category,'customer':customer,'checks':checks,'draft':None,'warnings':[checks.get('block_reason')], 'retrieved_sources':[]}
    try:
        from src.nexora_rag.pipeline import NexoraAssistant
        assistant=NexoraAssistant()
        prompt=req.input_text or req.objective or (
            f'Prepare a concise, personalised email for this customer. Use the verified profile, '
            f'predicted segment ({customer.get("segment_label")}), relevant recent activity and retrieved '
            f'policy context. Explain only verified facts; do not claim offer eligibility unless the '
            f'deterministic checks explicitly establish it. Category: {category}.'
        )
        result=assistant.answer(prompt, category, customer, checks)
        return {'status':'DRAFT_FOR_HUMAN_REVIEW','category':category,'customer':customer,'checks':checks,**result}
    except Exception as exc:
        raise HTTPException(503, f'RAG/LLM not ready. Confirm policy ingestion, model downloads and provider API key. {type(exc).__name__}: {exc}')

@app.post('/api/ingest')
def ingest():
    try:
        from src.nexora_rag.ingestion import ingest_policies
        return ingest_policies(reset=False)
    except Exception as exc: raise HTTPException(500, f'Policy ingestion failed: {type(exc).__name__}: {exc}')

FRONT=ROOT/'frontend'
app.mount('/',StaticFiles(directory=FRONT,html=True),name='frontend')
