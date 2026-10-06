const STEPS = [
  { title: 'Collect', text: 'Photograph good parts with the camera and lighting you will use on the line. A few defect photos help, but are not required to start.' },
  { title: 'Train on good parts', text: 'The anomaly model learns what "normal" looks like and flags anything different, including defect types nobody has seen before.' },
  { title: 'Evaluate honestly', text: 'A held-out test set produces the numbers: missed defects, false rejects, and the failure cases, as images.' },
  { title: 'Set the threshold', text: 'Choose where to draw the line using the real trade-off, optionally weighted by what a miss and a false reject cost you.' },
  { title: 'Inspect', text: 'Every part gets PASS, REVIEW or FAIL, with a heatmap showing where the problem is.' },
  { title: 'Review and improve', text: 'Borderline parts go to an operator. Their decisions become labels for the next training round.' },
];

const METHODS = [
  { situation: 'Plenty of good parts, few or no defect photos (most common)', method: 'Unsupervised anomaly detection (PatchCore, EfficientAD, PaDiM)', why: 'Learns normal and catches unknown defect types' },
  { situation: 'Hundreds of labelled examples per defect type, and the type matters', method: 'Supervised detection or segmentation', why: 'Names each defect class and outlines it' },
  { situation: 'Both', method: 'Hybrid: anomaly gate, then a classifier on flagged parts', why: 'Catches the unknown and sorts the known' },
  { situation: 'Only a handful of images', method: 'Few-shot / zero-shot check', why: 'A quick go / no-go, not a production model' },
];

const DELIVERABLES = [
  { title: 'Evaluation report', text: 'PDF with metrics, the threshold trade-off, heatmaps and the model’s misses. Written so a QA manager can forward it.' },
  { title: 'Inspection dashboard', text: 'Live verdicts, heatmaps, yield and defect-type breakdown, plus a review queue for borderline parts.' },
  { title: 'REST API + Docker', text: 'POST an image, get back verdict, score, heatmap and defect regions. Runs on a CPU industrial PC.' },
  { title: 'Edge optimisation', text: 'ONNX / OpenVINO export with measured speed on your target hardware, reported with the hardware named.' },
];

export default function ProductPage() {
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">How it works</p>
          <h1 className="display">From a folder of photos to a pass/fail inspector</h1>
          <p className="lead">
            Most factories have thousands of photos of good parts and very few of bad ones. Defecto starts from the good ones, then
            proves how well it works before anything touches your line.
          </p>
        </div>
        <a className="btn btn--dark" href="#/demo">
          Open the demo
        </a>
      </div>

      <section className="steps" aria-label="Workflow">
        {STEPS.map((s) => (
          <div key={s.title} className="card step">
            <p className="h3">{s.title}</p>
            <p>{s.text}</p>
          </div>
        ))}
      </section>

      <section style={{ marginTop: 48 }}>
        <h2 className="h2">The right method for your data</h2>
        <p className="lead" style={{ margin: '10px 0 18px' }}>
          The approach is chosen from what you actually have, not from what sounds impressive.
        </p>
        <div className="card table-wrap" style={{ padding: 6 }}>
          <table className="table">
            <thead>
              <tr>
                <th>Your situation</th>
                <th>Method</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {METHODS.map((m) => (
                <tr key={m.situation}>
                  <td>{m.situation}</td>
                  <td>{m.method}</td>
                  <td>{m.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section style={{ marginTop: 48 }}>
        <h2 className="h2">What you get</h2>
        <div className="grid-4" style={{ marginTop: 18 }}>
          {DELIVERABLES.map((d) => (
            <div key={d.title} className="card">
              <p className="h3">{d.title}</p>
              <p className="small" style={{ margin: '8px 0 0', color: 'var(--c-ink-soft)' }}>
                {d.text}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="card" style={{ marginTop: 48, padding: 24, display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <p className="h3">Not sure it will work on your parts?</p>
          <p className="small muted" style={{ margin: '6px 0 0' }}>
            That is exactly what the feasibility study answers, using your own images.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <a className="btn btn--light" href="#/pricing">
            See packages
          </a>
          <a className="btn btn--dark" href="#/pilot">
            Book a pilot
          </a>
        </div>
      </section>
    </div>
  );
}
