const PLANS = [
  {
    name: 'Feasibility Study',
    price: '$300',
    days: '5 days',
    pitch: '“Can AI inspect my part?” A go / no-go answer with evidence.',
    features: [
      'Up to 300 of your images, 1 product view',
      'Anomaly model trained on good parts',
      'PDF report: metrics, heatmaps, misses',
      'Clear recommendation and next-step quote',
      'Data collection guide',
      '1 revision',
    ],
  },
  {
    name: 'Pilot Inspector',
    price: '$950',
    days: '12 days',
    pitch: 'A working model and dashboard on your line’s images.',
    featured: true,
    features: [
      'Up to 3,000 images, up to 3 defect classes',
      'Anomaly or supervised model, tuned',
      'Threshold tuned to your cost of errors',
      'Inspection dashboard + REST API + Docker',
      '7 days of support after delivery',
      '2 revisions',
    ],
  },
  {
    name: 'Production Deployment',
    price: '$2,500',
    days: '21 days',
    pitch: 'An optimised edge system ready for the shop floor.',
    features: [
      'Up to 3 products or views, hybrid model',
      'ONNX / OpenVINO edge package, benchmarked',
      'Camera integration and review queue',
      'One line signal: MQTT, OPC UA or Modbus',
      'Shift reports and handover docs',
      '30 days of support, 3 revisions',
    ],
  },
];

const ADDONS = [
  ['Extra product or view', '+$250'],
  ['Extra defect class', '+$150'],
  ['Labelling, per 500 images', '+$60'],
  ['Monthly retraining and drift monitoring', 'Custom'],
];

const EXCLUDED = [
  'Buying cameras, lenses or lighting',
  'On-site installation',
  'Labelling large datasets (quoted separately)',
  'Safety or regulatory certification',
  'Accuracy guarantees before a feasibility study',
];

export default function PricingPage() {
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Pricing</p>
          <h1 className="display">Start small, prove it, then scale</h1>
          <p className="lead">
            Every project starts with measurable results on your own images. Each step has written acceptance criteria agreed before work
            begins.
          </p>
        </div>
      </div>

      <div className="plans">
        {PLANS.map((p) => (
          <div key={p.name} className={`card plan${p.featured ? ' plan--featured' : ''}`}>
            <p className="h3">{p.name}</p>
            <p className="plan__price">{p.price}</p>
            <p className="small muted" style={{ margin: 0 }}>
              Delivery in {p.days}
            </p>
            <p className="small" style={{ margin: '14px 0 0', lineHeight: 1.5 }}>
              {p.pitch}
            </p>
            <ul>
              {p.features.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
            <a className={`btn ${p.featured ? 'btn--accent' : 'btn--dark'}`} href={`#/pilot?plan=${encodeURIComponent(p.name)}`}>
              Choose {p.name}
            </a>
          </div>
        ))}
      </div>
      <p className="small muted" style={{ marginTop: 12 }}>
        Draft prices. Larger projects (several lines, on-premise servers, MES integration) are quoted as milestone offers.
      </p>

      <div className="grid-2" style={{ marginTop: 32 }}>
        <div className="card">
          <p className="h3">Add-ons</p>
          <table className="table" style={{ marginTop: 8 }}>
            <tbody>
              {ADDONS.map(([name, price]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="num" style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                    {price}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <p className="h3">Not included unless quoted</p>
          <ul className="small" style={{ margin: '12px 0 0', paddingLeft: 18, lineHeight: 1.8, color: 'var(--c-ink-soft)' }}>
            {EXCLUDED.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
