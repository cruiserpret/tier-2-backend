import type { DemoProduct } from "../types";

interface Props {
  demos: DemoProduct[];
  active: string | null;
  onSelect: (d: DemoProduct) => void;
  onClear: () => void;
}

export function DemoSelector({ demos, active, onSelect, onClear }: Props) {
  return (
    <div className="demo-chips-wrap fade-up fade-up-1">
      <div className="section-eyebrow">Quick demos — prefill the form</div>
      <div className="demo-chips">
        {demos.map((d) => (
          <button
            key={d.key}
            type="button"
            className={`chip ${active === d.key ? "active" : ""}`}
            onClick={() => onSelect(d)}
            title={d.story}
          >{d.label}</button>
        ))}
        <button
          type="button"
          className="chip chip-clear"
          onClick={onClear}
          title="Start with a blank form"
        >+ Custom</button>
      </div>
    </div>
  );
}
