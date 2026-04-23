import { useState, useEffect, useRef } from 'react';
import * as api from '../api';
import type { Variable, Scenario, ModelInfo } from '../api';

interface SettingsDrawerProps {
  open: boolean;
  onClose: () => void;
}

export default function SettingsDrawer({ open, onClose }: SettingsDrawerProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [variables, setVariables] = useState<Variable[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [datasets, setDatasets] = useState<api.Dataset[]>([]);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    closeBtnRef.current?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    Promise.all([
      api.listModels().catch(() => ({ models: [] })),
      api.listVariables().catch(() => ({ variables: [] })),
      api.listScenarios().catch(() => ({ scenarios: [] })),
      api.listDatasets().catch(() => ({ datasets: [], dataset_count: 0 })),
    ]).then(([m, v, s, d]) => {
      setModels((m as any).models_info?.slice(0, 15) || (m as any).models?.slice(0, 15).map((n: string) => ({ name: n, description: '' })) || []);
      setVariables((v as any).variables || []);
      setScenarios((s as any).scenarios || []);
      setDatasets((d as any).datasets || []);
    });
  }, [open]);

  return (
    <>
      {open && <div className="drawer-backdrop" onClick={onClose} />}
      <div
        className={`settings-drawer ${open ? 'open' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-drawer-title"
        aria-hidden={!open}
      >
        <div className="drawer-header">
          <h2 id="settings-drawer-title">Settings & Data</h2>
          <button
            ref={closeBtnRef}
            className="drawer-close"
            onClick={onClose}
            aria-label="Close settings"
          >
            &times;
          </button>
        </div>

        <div className="drawer-content">
          <section>
            <h3>Available Models</h3>
            <div className="drawer-list">
              {models.map(m => (
                <div key={m.name} className="drawer-list-item">
                  <strong>{m.name}</strong>{m.description && ` — ${m.description}`}
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3>Variables</h3>
            <div className="drawer-list">
              {variables.map(v => (
                <div key={v.name} className="drawer-list-item">
                  <strong>{v.name}</strong> - {v.long_name} ({v.units})
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3>Scenarios</h3>
            <div className="drawer-list">
              {scenarios.map(s => (
                <div key={s.id} className="drawer-list-item">
                  <strong>{s.id}</strong> - {s.description}
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3>Loaded Datasets ({datasets.length})</h3>
            {datasets.length === 0 ? (
              <p className="drawer-hint">No datasets loaded. Ask the AI to load data.</p>
            ) : (
              <div className="drawer-list">
                {datasets.map(ds => (
                  <div key={ds.id} className="drawer-list-item">
                    <code>{ds.id}</code>
                    <span>{ds.variable} / {ds.model} / {ds.scenario}</span>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </>
  );
}
