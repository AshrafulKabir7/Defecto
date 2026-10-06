const FAQ = [
  {
    q: 'Do I need photos of defective parts?',
    a: 'Not to start. Anomaly detection learns from good parts only. A few defect photos (even 5 to 10) are used to measure how well it works.',
  },
  {
    q: 'What accuracy will I get?',
    a: 'It depends on your parts, camera and lighting, so it is measured on your images in the feasibility study before you invest more. You get missed-defect and false-reject rates, not a headline percentage.',
  },
  {
    q: 'What is the REVIEW band?',
    a: 'Parts the model is unsure about go to a person instead of being passed or rejected automatically. It keeps missed defects low without scrapping lots of good parts, and the operator’s decisions become new training labels.',
  },
  {
    q: 'Can it run without the internet?',
    a: 'Yes. Models are exported to run locally on a CPU industrial PC or an edge GPU. Nothing needs to leave the factory network.',
  },
  {
    q: 'How fast is it?',
    a: 'Speed is measured on your target hardware and stated in the report with the hardware named. It is never guessed.',
  },
  {
    q: 'Will my images stay private?',
    a: 'Yes. They are used only for your project, deleted on request, and never shown in any portfolio without your written permission. An NDA is fine.',
  },
  {
    q: 'Can it connect to my PLC or MES?',
    a: 'Yes, as a scoped integration (MQTT, OPC UA or Modbus). The protocol and the equipment on your side are confirmed before quoting.',
  },
  {
    q: 'Is it suitable for safety-critical parts?',
    a: 'It can assist inspection, but final validation and certification remain with your quality system. It is not sold as a certified safety device.',
  },
  {
    q: 'Where do the demo images come from?',
    a: 'From the VisA dataset by Amazon Science (CC BY 4.0). Results on public data show the process; they are not a promise for your parts.',
  },
];

export default function FaqPage() {
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">FAQ</p>
          <h1 className="display">Straight answers</h1>
          <p className="lead">The questions quality teams ask before a pilot.</p>
        </div>
      </div>
      <div className="faq">
        {FAQ.map((f, i) => (
          <details key={f.q} open={i === 0}>
            <summary>{f.q}</summary>
            <p>{f.a}</p>
          </details>
        ))}
      </div>
      <div style={{ marginTop: 32, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <a className="btn btn--dark" href="#/pilot">
          Ask about your parts
        </a>
        <a className="btn btn--light" href="#/pricing">
          See pricing
        </a>
      </div>
    </div>
  );
}
