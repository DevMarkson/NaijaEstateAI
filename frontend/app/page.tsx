"use client";
import React, { useState } from 'react';

interface PredictResponse { prediction: number; rounded: number; currency: string }

const defaultPayload = {
  bedrooms: 3,
  bathrooms: 3,
  toilets: 4,
  Serviced: 0,
  Newly_Built: 0,
  Furnished: 0,
  property_type: 'Apartment',
  City: 'Lagos',
  Neighborhood: 'Lekki'
};

export default function Home() {
  const [form, setForm] = useState(defaultPayload);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');

  const update = (k: string, v: any) => setForm(f => ({ ...f, [k]: v }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      });
      if(!r.ok) throw new Error(`HTTP ${r.status}`);
      const data: PredictResponse = await r.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const numInput = (label: string, field: keyof typeof form, min=0, max=25) => (
    <label style={{display:'flex', flexDirection:'column', fontSize:14}}>
      {label}
      <input type="number" value={form[field] as any} min={min} max={max}
        onChange={e=> update(field, Number(e.target.value))}
        style={{padding:6, border:'1px solid #ccc', borderRadius:4}} />
    </label>
  );

  const boolSelect = (label: string, field: keyof typeof form) => (
    <label style={{display:'flex', flexDirection:'column', fontSize:14}}>
      {label}
      <select value={form[field] as any} onChange={e=> update(field, Number(e.target.value))}
        style={{padding:6, border:'1px solid #ccc', borderRadius:4}}>
        <option value={0}>No</option>
        <option value={1}>Yes</option>
      </select>
    </label>
  );

  return (
    <main style={{maxWidth:880, margin:'0 auto', padding:'2rem', fontFamily:'system-ui, sans-serif'}}>
      <h1 style={{marginTop:0}}>NaijaEstateAI – Rent Predictor</h1>
      <p style={{color:'#555'}}>Client UI calling FastAPI backend at <code>{apiUrl}</code>.</p>
      <form onSubmit={submit} style={{display:'grid', gap:'1rem', gridTemplateColumns:'repeat(auto-fill, minmax(180px,1fr))'}}>
        <label style={{gridColumn:'1 / -1'}}>
          API Base URL
          <input value={apiUrl} onChange={e=> setApiUrl(e.target.value)} style={{padding:6, width:'100%'}} />
        </label>
        {numInput('Bedrooms','bedrooms',0,20)}
        {numInput('Bathrooms','bathrooms',0,20)}
        {numInput('Toilets','toilets',0,25)}
        {boolSelect('Serviced','Serviced')}
        {boolSelect('Newly Built','Newly_Built')}
        {boolSelect('Furnished','Furnished')}
        <label style={{display:'flex', flexDirection:'column', fontSize:14}}>
          Property Type
          <input value={form.property_type} onChange={e=> update('property_type', e.target.value)} style={{padding:6}} />
        </label>
        <label style={{display:'flex', flexDirection:'column', fontSize:14}}>
          City
          <input value={form.City} onChange={e=> update('City', e.target.value)} style={{padding:6}} />
        </label>
        <label style={{display:'flex', flexDirection:'column', fontSize:14}}>
          Neighborhood
          <input value={form.Neighborhood} onChange={e=> update('Neighborhood', e.target.value)} style={{padding:6}} />
        </label>
        <div style={{gridColumn:'1 / -1', display:'flex', gap:12, alignItems:'center'}}>
          <button type="submit" disabled={loading} style={{padding:'0.7rem 1.2rem', background:'#0b5', color:'#fff', border:'none', borderRadius:6, cursor:'pointer'}}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
          {result && <span style={{fontWeight:600}}>Estimated: ₦{result.rounded.toLocaleString()}</span>}
          {error && <span style={{color:'crimson'}}>Error: {error}</span>}
        </div>
      </form>
      <section style={{marginTop:'2rem'}}>
        <h2>Sample cURL</h2>
        <pre style={{background:'#f5f5f5', padding:'1rem', overflowX:'auto'}}>{`curl -X POST ${apiUrl}/predict \
  -H 'Content-Type: application/json' \
  -d '${JSON.stringify(form, null, 0)}'`}</pre>
      </section>
    </main>
  );
}
