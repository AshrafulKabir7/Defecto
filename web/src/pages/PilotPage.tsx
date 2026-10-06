import { useState, type FormEvent } from 'react';

const PLANS = ['Not sure yet', 'Feasibility Study', 'Pilot Inspector', 'Production Deployment'];
const HARDWARE = ['Not decided', 'Industrial PC (CPU only)', 'PC with NVIDIA GPU', 'NVIDIA Jetson', 'Cloud'];
const ON_FAIL = ['Show on an operator screen', 'Signal a PLC / reject arm', 'Log to MES / database', 'Not decided'];

type Fields = {
  name: string;
  email: string;
  company: string;
  plan: string;
  product: string;
  defects: string;
  images: string;
  speed: string;
  hardware: string;
  onFail: string;
  notes: string;
};

function initialPlan() {
  const q = new URLSearchParams(window.location.hash.split('?')[1] ?? '');
  const plan = q.get('plan');
  return plan && PLANS.includes(plan) ? plan : PLANS[0];
}

export default function PilotPage() {
  const [f, setF] = useState<Fields>({
    name: '', email: '', company: '', plan: initialPlan(), product: '', defects: '',
    images: '', speed: '', hardware: HARDWARE[0], onFail: ON_FAIL[0], notes: '',
  });
  const [errors, setErrors] = useState<Partial<Record<keyof Fields, string>>>({});
  const [summary, setSummary] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const set = (k: keyof Fields) => (e: { target: { value: string } }) => {
    setF((s) => ({ ...s, [k]: e.target.value }));
    setErrors((er) => ({ ...er, [k]: undefined }));
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const next: Partial<Record<keyof Fields, string>> = {};
    if (!f.name.trim()) next.name = 'Please add your name.';
    if (!/^\S+@\S+\.\S+$/.test(f.email)) next.email = 'Please add a valid email.';
    if (f.product.trim().length < 3) next.product = 'What part should be inspected?';
    if (f.defects.trim().length < 3) next.defects = 'Which defects matter most?';
    setErrors(next);
    if (Object.keys(next).length) return;
    setSummary(
      [
        `Pilot request: ${f.plan}`,
        '',
        `Name: ${f.name}`,
        `Email: ${f.email}`,
        f.company && `Company: ${f.company}`,
        `Part / product: ${f.product}`,
        `Defects that matter: ${f.defects}`,
        f.images && `Images available: ${f.images}`,
        f.speed && `Line speed: ${f.speed}`,
        `Target hardware: ${f.hardware}`,
        `On a FAIL: ${f.onFail}`,
        f.notes && `Notes: ${f.notes}`,
      ]
        .filter(Boolean)
        .join('\n'),
    );
  };

  const field = (k: keyof Fields, label: string, el: JSX.Element, full = false) => (
    <label className={`field${full ? ' full' : ''}`} data-invalid={!!errors[k]}>
      {label}
      {el}
      {errors[k] && <span className="err">{errors[k]}</span>}
    </label>
  );

  return (
    <div className="page" style={{ maxWidth: 860 }}>
      <div className="page-head">
        <div>
          <p className="eyebrow">Book a pilot</p>
          <h1 className="display">Tell us about your parts</h1>
          <p className="lead">
            A short description is enough to say whether AI inspection is a good fit, and which package makes sense. Attach 10 to 20
            sample images when you send it.
          </p>
        </div>
      </div>

      {summary ? (
        <div className="card" style={{ padding: 24 }}>
          <p className="h3">Your request is ready</p>
          <p className="small muted" style={{ margin: '6px 0 14px' }}>
            Nothing is sent from this page. Open it in your email app or copy it into a Fiverr message, and attach sample images.
          </p>
          <pre className="panel small" style={{ padding: 16, whiteSpace: 'pre-wrap', margin: 0, fontFamily: 'inherit' }}>
            {summary}
          </pre>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 16 }}>
            <a className="btn btn--dark" href={`mailto:?subject=${encodeURIComponent(`Defecto pilot request: ${f.product}`)}&body=${encodeURIComponent(summary)}`}>
              Open in email app
            </a>
            <button
              type="button"
              className="btn btn--light"
              onClick={() => {
                navigator.clipboard
                  .writeText(summary)
                  .then(() => {
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1800);
                  })
                  .catch(() => {});
              }}
            >
              {copied ? 'Copied ✓' : 'Copy request'}
            </button>
            <button type="button" className="btn btn--light" onClick={() => setSummary(null)}>
              Edit
            </button>
          </div>
        </div>
      ) : (
        <form className="card form" style={{ padding: 24 }} onSubmit={submit} noValidate>
          {field('name', 'Your name', <input value={f.name} onChange={set('name')} autoComplete="name" />)}
          {field('email', 'Email', <input type="email" value={f.email} onChange={set('email')} autoComplete="email" />)}
          {field('company', 'Company (optional)', <input value={f.company} onChange={set('company')} autoComplete="organization" />)}
          {field(
            'plan',
            'Package',
            <select value={f.plan} onChange={set('plan')}>
              {PLANS.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>,
          )}
          {field('product', 'What part should be inspected?', <input value={f.product} onChange={set('product')} placeholder="e.g. injection-moulded bottle caps" />, true)}
          {field('defects', 'Which defects matter most?', <textarea rows={3} value={f.defects} onChange={set('defects')} placeholder="e.g. cracks, short shots, black specks. Which are critical vs cosmetic?" />, true)}
          {field('images', 'Images available (optional)', <input value={f.images} onChange={set('images')} placeholder="e.g. 500 good, 20 defective" />)}
          {field('speed', 'Line speed (optional)', <input value={f.speed} onChange={set('speed')} placeholder="e.g. 60 parts per minute" />)}
          {field(
            'hardware',
            'Target hardware',
            <select value={f.hardware} onChange={set('hardware')}>
              {HARDWARE.map((h) => (
                <option key={h}>{h}</option>
              ))}
            </select>,
          )}
          {field(
            'onFail',
            'What should happen on a FAIL?',
            <select value={f.onFail} onChange={set('onFail')}>
              {ON_FAIL.map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>,
          )}
          {field('notes', 'Anything else (optional)', <textarea rows={3} value={f.notes} onChange={set('notes')} placeholder="Camera and lighting, NDA needs, deadlines..." />, true)}
          <div className="full" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <button type="submit" className="btn btn--dark">
              Prepare request
            </button>
            <span className="small muted">Your details stay in this browser until you send them yourself.</span>
          </div>
        </form>
      )}
    </div>
  );
}
